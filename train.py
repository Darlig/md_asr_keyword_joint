import argparse
import copy
import os
import re

import torch
import torch.distributed as dist
import torch.optim as optim
import yaml

from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from yamlinclude import YamlIncludeConstructor

from data.loader.data_loader import Dataset
from local.utils import Recorder, WarmUpLR, read_list


TRAINABLE_PRESETS = {
    "all": {"freeze_prefixes": []},
    "freeze_phone": {"freeze_prefixes": ["phn_emb", "kw_", "md_transformer", "det_net", "det_nets"]},
    "freeze_speech": {"freeze_prefixes": ["hubert_", "au_", "phn_asr_crit"]},
    "phone_only": {"train_prefixes": ["phn_emb", "kw_"]},
}


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="config file in yaml format e.g. config/ref.yaml")
    parser.add_argument("--step", required=False, default=0, type=int, help="step")
    parser.add_argument("--world_size", required=True, type=int, help="world size")
    parser.add_argument("--rank", required=True, type=int, help="rank")
    parser.add_argument("--port", required=False, default="1234", type=str, help="port")
    parser.add_argument("--gpu", required=True, help="gpu id")
    parser.add_argument("--seed", default=2022, help="random seed")
    parser.add_argument("--exp_root", default="exp/", help="experiments root dir")
    return parser.parse_args()


def move_to_device(x, device):
    if torch.is_tensor(x):
        return x.to(device, non_blocking=True)
    if isinstance(x, (list, tuple)):
        return type(x)(move_to_device(v, device) for v in x)
    if isinstance(x, dict):
        return {k: move_to_device(v, device) for k, v in x.items()}
    return x


class Trainer:
    def __init__(self, model_arch, config_file: dict, rank: int, world_size: int, random_seed=2022):
        self.config_file = config_file
        self.data_config = config_file["data_config"]
        self.valid_data_config = config_file["data_config"].copy()
        self.valid_data_config.update({"batch_size": 32})
        self.exp_config = config_file["exp_config"]

        self.model_config = copy.deepcopy(config_file["model_config"])
        self._inject_input_mode_from_data_config()
        self.model = model_arch(**self.model_config)

        self.rank = rank
        self.seed = random_seed
        self.device = torch.device("cuda")
        self.world_size = world_size

        if (not os.path.isdir(self.exp_config["exp_dir"])) and (self.rank == 0):
            try:
                os.makedirs(self.exp_config["exp_dir"])
            except Exception as exc:
                raise FileNotFoundError(
                    f"can not create exp dir: {self.exp_config['exp_dir']}"
                ) from exc

        self.exp_config["log_config"]["filename"] = "{}/train.{}.log".format(
            self.exp_config["exp_dir"],
            self.rank,
        )
        self.recorder = Recorder(self.exp_config)

    def _inject_input_mode_from_data_config(self):
        input_mode = self.model_config.get("input_mode", "auto")
        if input_mode != "auto":
            return

        sph_config = self.data_config.get("sph_config", {})
        data_type = sph_config.get("data_type", "raw")
        if data_type == "raw":
            inferred = "waveform"
        elif data_type == "torch":
            inferred = "embedding"
        else:
            raise ValueError(f"Unsupported sph_config.data_type for auto input mode: {data_type}")
        self.model_config["input_mode"] = inferred

    def backup_configs(self):
        for k, v in self.config_file.items():
            self.recorder.info("{} config : {}".format(k.upper(), v))

        if (self.rank == 0) and (self.data_config["start_epoch"] == 0):
            with open("{}/model.yaml".format(self.exp_config["exp_dir"]), "w") as mf:
                yaml.dump(self.model_config, mf)
            with open("{}/data.yaml".format(self.exp_config["exp_dir"]), "w") as df:
                yaml.dump(self.data_config, df)
            with open("{}/exp.yaml".format(self.exp_config["exp_dir"]), "w") as ef:
                yaml.dump(self.exp_config, ef)

    def compute_redundancy(self, n, batch_size):
        r1 = n % self.world_size
        r2 = ((n - r1) / self.world_size) % batch_size
        rt = n - r1 - r2 * self.world_size
        return int(rt)

    def make_data_loader(self):
        data_list_file = self.data_config["data_list"]
        self.batch_size = self.data_config["batch_size"]
        self.valid_batch_size = self.valid_data_config["batch_size"]
        cv_list_file = self.data_config.get("valid_list", None)

        if cv_list_file:
            cv_list = read_list(cv_list_file, split_cv=False, shuffle=True)
            tr_list = read_list(data_list_file, split_cv=False, shuffle=True)
        else:
            tr_list, cv_list = read_list(data_list_file, split_cv=True, shuffle=True)

        rt_train_sample = self.compute_redundancy(len(tr_list), self.batch_size)
        rt_cv_sample = self.compute_redundancy(len(cv_list), self.valid_batch_size)

        tr_list = tr_list[:rt_train_sample]
        cv_list = cv_list[:rt_cv_sample]

        if self.data_config.get("egs_format", False):
            egs_path = os.path.dirname(data_list_file)
            assert os.path.isfile("{}/train.samples".format(egs_path))
            with open("{}/train.samples".format(egs_path), "r") as ef:
                self.num_samples = int(ef.readline().strip())
        else:
            self.num_samples = len(tr_list)
        num_worker = self.data_config.get("num_worker", 10)

        self.tr_set = Dataset(self.data_config, tr_list)
        self.cv_set = Dataset(self.valid_data_config, cv_list)

        self.tr_loader = DataLoader(self.tr_set, batch_size=None, num_workers=num_worker)
        self.cv_loader = DataLoader(self.cv_set, batch_size=None, num_workers=3)

        if self.data_config["start_epoch"] == 0:
            self.recorder.info(
                "Num Training samples: {}, Num Valid samples: {}".format(len(tr_list), len(cv_list))
            )
            self.recorder.info("Num Worker: {}".format(num_worker))

    def init_opt_model(self):
        self.batch_size = self.data_config["batch_size"]
        steps_per_epoch = self.num_samples // (self.batch_size * self.world_size)
        steps_per_epoch = 1 if steps_per_epoch == 0 else steps_per_epoch
        warm_up_peak_epoch = self.exp_config.get("warm_up_peak_epoch", 5)
        warm_up_peak_step = warm_up_peak_epoch * steps_per_epoch

        num_param = sum([v.numel() for v in self.model.parameters()])
        self.clip_value = self.exp_config.get("clip_value", 10.0)
        self.optim = optim.Adam(self.model.parameters(), **self.exp_config["optim_config"])

        start_epoch = self.data_config.get("start_epoch", 0)
        self.global_step = 0
        if start_epoch != 0:
            ckpt = self.load_endpoint(start_epoch - 1)
            resume_optimizer = self.exp_config.get(
                "resume_optimizer",
                self.exp_config.get("trainable_mode", "all") == "all",
            )
            self.global_step = self.load_checkpoint(ckpt, load_optimizer=resume_optimizer)

        self.scheduler = WarmUpLR(self.optim, warmup_steps=warm_up_peak_step)
        self.scheduler.set_step(self.global_step)

        if self.exp_config.get("finetune", False):
            self.init_from_trained(**self.exp_config.get("finetune"))

        self.apply_trainable_mode(self.exp_config.get("trainable_mode", "all"))

        if self.world_size > 1:
            dist.init_process_group("nccl", world_size=self.world_size, rank=self.rank)
            self.model.cuda()
            self.model = torch.nn.parallel.DistributedDataParallel(self.model)
        else:
            self.model.to(self.device)

        if start_epoch == 0:
            self.recorder.info("WarmUp epoch: {} WarmUp steps:{}".format(warm_up_peak_epoch, warm_up_peak_step))
            self.recorder.info("Gradient Clip Value: {}".format(self.clip_value))
            self.recorder.info("Number parameter: {}".format(num_param))
        else:
            self.recorder.info("Continue training from epoch: {}".format(start_epoch))

    def init_from_trained(self, trained_ckpt, percentage_fix_layer="last"):
        self.recorder.info(
            "Init from trained checkpoint {} {}\% of them will be fixed".format(
                trained_ckpt,
                percentage_fix_layer,
            )
        )
        trained_ckpt = torch.load(trained_ckpt, map_location="cpu")
        trained_model = trained_ckpt["model"]
        trained_keys = list(trained_model.keys())
        fix_component = []
        current_state_dict = self.model.state_dict()
        num_trained_layers = len(trained_keys)
        if isinstance(percentage_fix_layer, int) and (percentage_fix_layer > 1):
            percentage_fix_layer = float(percentage_fix_layer) / 100
        num_fix_layer = (
            num_trained_layers - 2
            if percentage_fix_layer == "last"
            else float(percentage_fix_layer) * num_trained_layers
        )
        self.recorder.info("load parameters from trained model: {}".format(trained_ckpt.keys()))
        for i, k in enumerate(trained_keys):
            trained_param = trained_model[k]
            if (current_state_dict[k].size() == trained_param.size()) and (i < num_fix_layer):
                fix_component.append(k)
            else:
                trained_model.pop(k)
        self.model.load_state_dict(trained_model, strict=False)
        for name, param in self.model.named_parameters():
            if name in fix_component:
                param.requires_grad = False
            else:
                self.recorder.info("{} are trainable".format(name))

    def _get_base_model(self):
        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            return self.model.module
        return self.model

    def _strip_module_prefix(self, state_dict):
        if any(k.startswith("module.") for k in state_dict.keys()):
            return {k[len("module."):]: v for k, v in state_dict.items()}
        return state_dict

    def _adapt_detection_heads(self, state_dict):
        model_state = self._get_base_model().state_dict()
        target_has_multi = any(k.startswith("det_nets.") for k in model_state.keys())
        ckpt_has_multi = any(k.startswith("det_nets.") for k in state_dict.keys())
        ckpt_has_single = any(k.startswith("det_net.") for k in state_dict.keys())

        if target_has_multi and ckpt_has_single and not ckpt_has_multi:
            head_ids = sorted(
                {
                    int(m.group(1))
                    for k in model_state.keys()
                    for m in [re.match(r"det_nets\.(\d+)\.", k)]
                    if m
                }
            )
            new_state = {k: v for k, v in state_dict.items() if not k.startswith("det_net.")}
            for k, v in state_dict.items():
                if not k.startswith("det_net."):
                    continue
                suffix = k[len("det_net.") :]
                for head_id in head_ids:
                    new_state[f"det_nets.{head_id}.{suffix}"] = v
            return new_state

        if (not target_has_multi) and ckpt_has_multi and ("det_net.0.w1.weight" not in model_state):
            new_state = {k: v for k, v in state_dict.items() if not k.startswith("det_nets.")}
            for k, v in state_dict.items():
                match = re.match(r"det_nets\.0\.(.+)", k)
                if match:
                    new_state[f"det_net.{match.group(1)}"] = v
            return new_state

        return state_dict

    def load_checkpoint(self, ckpt, load_optimizer=True):
        ckpt_dict = torch.load(ckpt, map_location="cpu")
        model = self._strip_module_prefix(ckpt_dict["model"])
        model = self._adapt_detection_heads(model)
        step = ckpt_dict["step"]

        incompatible_keys = self._get_base_model().load_state_dict(model, strict=False)
        if incompatible_keys.missing_keys:
            self.recorder.info("Missing keys: {}".format(incompatible_keys.missing_keys))
        if incompatible_keys.unexpected_keys:
            self.recorder.info("Unexpected keys: {}".format(incompatible_keys.unexpected_keys))

        if load_optimizer:
            opt = ckpt_dict["opt"]
            self.optim.load_state_dict(opt)
            for state in self.optim.state.values():
                for k, v in state.items():
                    if k == "step":
                        continue
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(self.device)
        return step

    def apply_trainable_mode(self, mode_name):
        if mode_name not in TRAINABLE_PRESETS:
            raise ValueError(
                f"Unsupported trainable_mode: {mode_name}. Supported: {sorted(TRAINABLE_PRESETS.keys())}"
            )
        preset = TRAINABLE_PRESETS[mode_name]
        freeze_prefixes = preset.get("freeze_prefixes", [])
        train_prefixes = preset.get("train_prefixes", [])

        for _, param in self._get_base_model().named_parameters():
            param.requires_grad = True

        for name, param in self._get_base_model().named_parameters():
            if freeze_prefixes and any(name.startswith(prefix) for prefix in freeze_prefixes):
                param.requires_grad = False
            if train_prefixes:
                param.requires_grad = any(name.startswith(prefix) for prefix in train_prefixes)

        self.recorder.info(f"Applied trainable_mode: {mode_name}")
        for name, param in self._get_base_model().named_parameters():
            if param.requires_grad:
                self.recorder.info(f"TRAINABLE: {name}")

    @torch.no_grad()
    def cross_valid(self):
        cv_model = copy.deepcopy(self.model)
        cv_model.eval()
        cv_detail_loss = {"total_loss": 0}
        num_utt = 0
        for cv_data in self.cv_loader:
            n = cv_data[0].size(0)
            cv_data = (move_to_device(d, self.device) for d in cv_data)
            num_utt += n
            total_loss, detail_loss = cv_model(cv_data)
            detail_loss = self.detach_from_graph(detail_loss)
            cv_detail_loss["total_loss"] += self.detach_from_graph(total_loss) * n

            for key, value in detail_loss.items():
                if key not in cv_detail_loss.keys():
                    cv_detail_loss[key] = value * n
                else:
                    cv_detail_loss[key] += value * n

        return {key: value / num_utt for key, value in cv_detail_loss.items()}

    def detach_from_graph(self, para):
        if isinstance(para, torch.Tensor):
            para = para.detach().clone()
        if isinstance(para, dict):
            para = {k: v.detach().clone() for k, v in para.items()}
        return para

    def detach_state_dict(self):
        d_model = copy.deepcopy(self.model)
        opt = copy.deepcopy(self.optim)
        if isinstance(d_model, torch.nn.parallel.DistributedDataParallel):
            d_model = d_model.module.state_dict()
        else:
            d_model = d_model.state_dict()
        return d_model, opt.state_dict()

    def _get_keep_last_ckpt_n(self):
        n = self.exp_config.get("keep_last_ckpt", self.exp_config.get("avg_epoch", 11))
        try:
            n = int(n)
        except Exception:
            n = 11
        return max(n, 1)

    def cleanup_epoch_checkpoints(self, keep_n: int, keep_epochs=None):
        keep_epochs = set() if keep_epochs is None else set(keep_epochs)
        exp_dir = self.exp_config["exp_dir"]
        exp_name = self.exp_config["exp_name"]
        pattern = re.compile(rf"^{re.escape(exp_name)}_(\d+)\.pt$")

        try:
            files = os.listdir(exp_dir)
        except Exception as exc:
            self.recorder.info(f"[ckpt-cleanup] failed to list dir {exp_dir}: {exc}")
            return

        epoch_files = []
        for fn in files:
            matched = pattern.match(fn)
            if not matched:
                continue
            epoch_files.append((int(matched.group(1)), os.path.join(exp_dir, fn)))

        if not epoch_files:
            return

        epoch_files.sort(key=lambda x: x[0])
        keep_set = set(ep for ep, _ in epoch_files[-keep_n:]) | keep_epochs
        to_delete = [(ep, path) for ep, path in epoch_files if ep not in keep_set]
        if not to_delete:
            return

        self.recorder.info(
            "[ckpt-cleanup] deleting {} old checkpoints; keeping newest {} (+{})".format(
                len(to_delete), keep_n, len(keep_epochs)
            )
        )
        for ep, path in to_delete:
            try:
                os.remove(path)
                self.recorder.info(f"[ckpt-cleanup] deleted epoch {ep}: {path}")
            except FileNotFoundError:
                continue
            except Exception as exc:
                self.recorder.info(f"[ckpt-cleanup] failed to delete epoch {ep}: {path} ({exc})")

    def record_step(self, r_loss):
        for key, value in r_loss.items():
            assert key in ["train", "cv"]
            self.recorder.record_detail(value, self.epoch, self.global_step, model=None, opt=None, tag=key)

    def record_epoch(self, cv_loss=None):
        model, opt = self.detach_state_dict()
        self.recorder.record_epoch(self.epoch, self.global_step, model, opt, cv_loss)

    def load_endpoint(self, epoch):
        ckpt = "{}/{}_{}.pt".format(
            self.exp_config["exp_dir"],
            self.exp_config["exp_name"],
            epoch,
        )
        if os.path.isfile(ckpt):
            return ckpt
        raise FileNotFoundError(
            "{} does not exits check the start epoch from yaml config file".format(ckpt)
        )

    def avg_model(self):
        max_epoch = self.data_config["epoch"]
        avg_epoch = self.exp_config.get("avg_epoch", 11)
        min_epoch = max_epoch - avg_epoch
        valid_ckpt = {k: 0 for k in range(min_epoch, max_epoch)}
        valid_loss = []
        for e in range(min_epoch, max_epoch):
            ckpt = "{}/{}_{}.pt".format(self.exp_config["exp_dir"], self.exp_config["exp_name"], e)
            ckpt = torch.load(ckpt, map_location="cpu")
            valid_ckpt[e] = ckpt["model"]
            valid_loss.append(ckpt["cv_loss"]["total_loss"].item())
        min_idx = sorted(range(len(valid_loss)), key=lambda k: valid_loss[k])[:avg_epoch]
        avg_model = None
        for idx in min_idx:
            epoch_id = idx + min_epoch
            state = valid_ckpt[epoch_id]
            if avg_model is None:
                avg_model = state
            else:
                for key in avg_model.keys():
                    avg_model[key] += state[key]
        for key in avg_model.keys():
            if avg_model[key] is not None:
                avg_model[key] = torch.true_divide(avg_model[key], avg_epoch)
        avg_tag = f"avg_{min_epoch}-{max_epoch-1}"
        self.recorder.save_state(avg_model, epoch=avg_tag)

    def train(self):
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(args.seed)
        self.model.to(self.device)
        self.model.train()
        start_epoch = self.data_config["start_epoch"]
        end_epoch = self.data_config["epoch"]
        self.recorder.info(
            "Start training the log is written in {}".format(self.exp_config["log_config"]["filename"])
        )

        for epoch in range(start_epoch, end_epoch):
            torch.cuda.empty_cache()
            self.epoch = epoch
            self.tr_set.set_epoch(epoch)
            for batch_id, data in enumerate(self.tr_loader):
                torch.cuda.empty_cache()
                tr_record_dict = {"lr": self.optim.param_groups[0]["lr"]}
                train_data = (move_to_device(d, self.device) for d in data)
                loss, detail_loss = self.model(train_data)
                self.optim.zero_grad()
                loss.backward()
                grad_norm = clip_grad_norm_(self.model.parameters(), self.clip_value)
                if torch.isfinite(grad_norm):
                    self.optim.step()
                else:
                    self.recorder.info("!!! INFINITE grad in epoch: {}, batch_id: {}".format(epoch, batch_id))
                self.scheduler.step()
                self.global_step += 1
                tr_record_dict["total_loss"] = loss
                tr_record_dict.update(detail_loss)
                if self.rank == 0:
                    self.record_step({"train": tr_record_dict})

            cv_record_dict = self.cross_valid()
            if self.rank == 0:
                self.record_step({"cv": cv_record_dict})
                self.record_epoch(cv_record_dict)
                self.cleanup_epoch_checkpoints(keep_n=self._get_keep_last_ckpt_n())

    def run(self, step):
        if step <= 0:
            self.backup_configs()
        if step <= 1:
            self.make_data_loader()
            self.init_opt_model()
            self.train()
        if (step <= 2) and (self.rank == 0):
            self.avg_model()
            last_epoch = int(self.data_config["epoch"]) - 1
            self.cleanup_epoch_checkpoints(keep_n=1, keep_epochs={last_epoch})


if __name__ == "__main__":
    args = get_args()
    YamlIncludeConstructor.add_to_loader_class(loader_class=yaml.FullLoader)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = args.port
    config = yaml.load(open(args.config), Loader=yaml.FullLoader)
    from model import m_dict

    model_arch = config["model_arch"]
    model = m_dict[model_arch]
    trainer = Trainer(model, config, world_size=args.world_size, rank=args.rank)
    trainer.run(args.step)
