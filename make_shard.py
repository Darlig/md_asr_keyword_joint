import json
import os
import tarfile
import sys

datalist = sys.argv[1]
out_dir = sys.argv[2]
out_map = sys.argv[3]

os.makedirs(out_dir, exist_ok=True)

shard_size = 100
index_map = {}

shard_id = 0
count = 0
tar = None

def open_new_tar(shard_id):
    path = os.path.join(out_dir, f"shard_{shard_id:05d}.tar")
    return tarfile.open(path, "w"), path

with open(datalist, "r") as f:
    for line in f:
        sample = json.loads(line)
        key = sample["key"]
        wav_path = sample["sph"]

        if count % shard_size == 0:
            if tar is not None:
                tar.close()
            tar, shard_path = open_new_tar(shard_id)
            shard_id += 1

        # 用 key 作为 tar member 名
        tar.add(wav_path, arcname=key)

        index_map[key] = {
            "shard": shard_path,
            "member": key
        }

        count += 1
print(len(index_map))
if tar is not None:
    tar.close()

with open(out_map, "w") as f:
    json.dump(index_map, f)

print(f"Done: {shard_id} shards, {len(index_map)} samples")
