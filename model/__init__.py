#__all__ = ['DoubleAE']
# new model append in m_dict
from model import AEDKWSASR
from model import AEDKWSASRPhone
from model import AEDASR
from model import AEDKWSASRPhoneUnet
from model import TransformerKWSPhone
from model import TransformerKWSPhone_hubert_wenet

m_dict = {
    'AEDKWSASR': AEDKWSASR.AEDKWSASR,
    'AEDKWSASRPhone': AEDKWSASRPhone.AEDKWSASRPhone,
    'AEDASR': AEDASR.AEDASR,
    'AEDKWSASRPhoneUnet': AEDKWSASRPhoneUnet.AEDKWSASRPhoneUnet,
    'TransformerKWSPhone': TransformerKWSPhone.TransformerKWSPhone,
    'TransformerKWSPhone_hubert_wenet': TransformerKWSPhone_hubert_wenet.TransformerKWSPhone_hubert_wenet,
    'TransformerKWSPhone_hubert_wenet_3md_adaptor': TransformerKWSPhone_hubert_wenet.TransformerKWSPhone_hubert_wenet,
    'TransformerKWSPhone_hubert_wenet_embed_3md_adaptor': TransformerKWSPhone_hubert_wenet.TransformerKWSPhone_hubert_wenet,
    'TransformerKWSPhone_hubert_wenet_embed': TransformerKWSPhone_hubert_wenet.TransformerKWSPhone_hubert_wenet,
}
