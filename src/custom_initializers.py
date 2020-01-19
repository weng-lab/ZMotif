import requests
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import MinMaxScaler

def svd():
    url = "http://hocomoco11.autosome.ru/final_bundle/hocomoco11/core/HUMAN/mono/HOCOMOCOv11_core_pwms_HUMAN_mono.txt"
    data = requests.get(url).text
    with open("motifs.txt","w") as f:
        f.write(data)

    PWMS_LIST = []
    with open("motifs.txt") as f:
        lines = [line.strip().split() for line in f.readlines()]

    pwm = []
    motif_ids = []
    for i, line in enumerate(lines):
        if line[0][0] == ">":
            motif_id = line[0][1:]
            if i > 0:
                pwm = np.reshape(np.array(pwm), (-1,4))
                PWMS_LIST.append(pwm)
                motif_ids.append(motif_id)
                pwm = []
        else:
            pwm.append([float(x) for x in line])

    #PWMS_LIST.append(np.array(pwm))
    max_w = np.max(np.array([pwm.shape[0] for pwm in PWMS_LIST]))
    k = len(PWMS_LIST)
    print(k)

    x = np.zeros((max_w*4,k))
    for i in range(k):
        pwm = PWMS_LIST[i]
    #    max_val = np.sum(np.max(pwm, axis=1))
        w = pwm.shape[0]
        start_index = np.random.randint(0,4*max_w - 4*w + 1)
        x[start_index:start_index+4*w,i] = pwm.flatten()

    svd = TruncatedSVD(n_components=32)
    svd.fit(x)
    result = svd.transform(x)
    scaler = MinMaxScaler(feature_range=(-0.01, 0.01), copy=True).fit(result)
    result = scaler.transform(result)
    conv_weights = np.reshape(result,(24,4,-1))
    return(conv_weights)
    