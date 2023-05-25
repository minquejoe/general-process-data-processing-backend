from np_4_2 import statistic_, time_dom_, freq_dom_, time_freq_dom_
from np_4_2_tsf import TSF_Data, TST_LSTM, TSF_ELM, TSF_SVR
from flask import Flask, request, jsonify
from flask_cors import *

app = Flask(__name__)
CORS(app, supports_credentials=True)


@app.route("/statistic_/get_max", methods=["POST"])
def get_max():
    data = request.get_json()["data"]
    output = statistic_(data).get_max()
    output = output.tolist()
    return jsonify(output)


@app.route("/statistic_/get_min", methods=["POST"])
def get_min():
    data = request.get_json()["data"]
    output = statistic_(data).get_min()
    output = output.tolist()
    return jsonify(output)


@app.route("/statistic_/get_avg", methods=["POST"])
def get_avg():
    data = request.get_json()["data"]
    output = statistic_(data).get_avg()
    output = output.tolist()
    return jsonify(output)


@app.route("/statistic_/get_mid", methods=["POST"])
def get_mid():
    data = request.get_json()["data"]
    output = statistic_(data).get_mid()
    return jsonify(output)


@app.route("/statistic_/get_len", methods=["POST"])
def get_len():
    data = request.get_json()["data"]
    output = statistic_(data).get_len()
    return jsonify(output)


@app.route("/statistic_/get_sum", methods=["POST"])
def get_sum():
    data = request.get_json()["data"]
    output = statistic_(data).get_sum()
    output = output.tolist()
    return jsonify(output)


@app.route("/statistic_/get_cup", methods=["POST"])
def get_cup():
    data = request.get_json()["data"]
    output = statistic_(data).get_cup()
    output = output.tolist()
    return jsonify(output)


@app.route("/statistic_/get_var", methods=["POST"])
def get_var():
    data = request.get_json()["data"]
    output = statistic_(data).get_var()
    output = output.tolist()
    return jsonify(output)


@app.route("/statistic_/get_std", methods=["POST"])
def get_std():
    data = request.get_json()["data"]
    output = statistic_(data).get_std()
    output = output.tolist()
    return jsonify(output)


@app.route("/statistic_/get_mnt", methods=["POST"])
def get_mnt():
    data = request.get_json()["data"]
    try:
        order = request.get_json()["order"]
    except:
        order = 3
    output = statistic_(data).get_mnt(int(order))
    output = output.tolist()
    return jsonify(output)


@app.route("/statistic_/get_pp", methods=["POST"])
def get_pp():
    data = request.get_json()["data"]
    output = statistic_(data).get_pp()
    output = output.tolist()
    return jsonify(output)


@app.route("/statistic_/get_ppc", methods=["POST"])
def get_ppc():
    data = request.get_json()["data"]
    output = statistic_(data).get_ppc()
    if output == "null":
        output = output.tolist()
    else:
        pass
    return jsonify(output)


@app.route("/statistic_/get_qtl", methods=["POST"])
def get_qtl():
    data = request.get_json()["data"]
    try:
        interval = request.get_json()["interval"]
    except:
        interval = 10
    output = statistic_(data).get_qtl(int(interval))
    output = output.tolist()
    return jsonify(output)


@app.route("/time_dom_/mul_win", methods=["POST"])
def mul_win():
    data = request.get_json()["data"]
    params = {
        "win_name": "hamming",
        "std": 0.65,
    }
    for key in params.keys():
        try:
            value_type = type(params[key])
            params[key] = value_type(request.get_json()[key])
        except:
            pass
    output = time_dom_(data).mul_win(**params)
    output = output.tolist()
    return jsonify(output)


@app.route("/time_dom_/spt_frm", methods=["POST"])
def spt_frm():
    data = request.get_json()["data"]
    params = {
        "fame_len": 256,
        "overlap_rate": float(0),
        "drop_last": 1,
    }
    for key in params.keys():
        try:
            value_type = type(params[key])
            params[key] = value_type(request.get_json()[key])
        except:
            pass
    output = time_dom_(data).spt_frm(**params)
    output = output.tolist()
    return jsonify(output)


@app.route("/time_dom_/sld_avg", methods=["POST"])
def sld_avg():
    data = request.get_json()["data"]
    params = {
        "scale": 4,
    }
    for key in params.keys():
        try:
            value_type = type(params[key])
            params[key] = value_type(request.get_json()[key])
        except:
            pass
    output = time_dom_(data).sld_avg(**params)
    output = output.tolist()
    return jsonify(output)


@app.route("/time_dom_/get_cov", methods=["POST"])
def get_cov():
    data = request.get_json()["data"]
    params = {
        "cov_with": [0.0],
        "mode": "same",
    }
    for key in params.keys():
        try:
            params[key] = request.get_json()[key]
        except:
            pass
    temp = params["cov_with"]
    params["cov_with"] = [float(x) for x in temp.split(",")]
    output = time_dom_(data).get_cov(**params)
    output = output.tolist()
    return jsonify(output)


@app.route("/time_dom_/up_samp", methods=["POST"])
def up_samp():
    data = request.get_json()["data"]
    params = {
        "ins_rate": 1,
        "mode": "zero",
    }
    for key in params.keys():
        try:
            value_type = type(params[key])
            params[key] = value_type(request.get_json()[key])
        except:
            pass
    output = time_dom_(data).up_samp(**params)
    output = output.tolist()
    return jsonify(output)


@app.route("/time_dom_/down_samp", methods=["POST"])
def down_samp():
    data = request.get_json()["data"]
    params = {
        "gap": 1,
    }
    for key in params.keys():
        try:
            value_type = type(params[key])
            params[key] = value_type(request.get_json()[key])
        except:
            pass
    output = time_dom_(data).down_samp(**params)
    output = output.tolist()
    return jsonify(output)


@app.route("/time_dom_/crs_crr", methods=["POST"])
def crs_crr():
    data = request.get_json()["data"]
    params = {
        "target": [0.0],
    }
    for key in params.keys():
        try:
            params[key] = request.get_json()[key]
        except:
            pass
    temp = params["target"]
    params["target"] = [float(x) for x in temp.split(",")]
    output = time_dom_(data).crs_crr(**params)
    return jsonify(output)


@app.route("/freq_dom_/get_dct", methods=["POST"])
def get_dct():
    data = request.get_json()["data"]
    params = {}
    for key in params.keys():
        try:
            value_type = type(params[key])
            params[key] = value_type(request.get_json()[key])
        except:
            pass
    output = freq_dom_(data).get_dct(**params)
    output = output.tolist()
    return jsonify(output)


@app.route("/freq_dom_/get_gabor", methods=["POST"])
def get_gabor():
    data = request.get_json()["data"]
    params = {
        "fs": 256,
        "std": 0.65,
        "nperseg": 256,
        "noverlap": 128,
    }
    for key in params.keys():
        try:
            value_type = type(params[key])
            params[key] = value_type(request.get_json()[key])
        except:
            pass
    output = freq_dom_(data).get_gabor(**params)
    return jsonify(output)


@app.route("/freq_dom_/dig_filt", methods=["POST"])
def dig_filt():
    data = request.get_json()["data"]
    params = {
        "filt_type": "lowpass",
        "fs": 256,
        "fc1": 16,
        "fc2": 32,
        "filter_od": 8,
    }
    for key in params.keys():
        try:
            value_type = type(params[key])
            params[key] = value_type(request.get_json()[key])
        except:
            pass
    if (params["filt_type"] == "bandpass") or (params["filt_type"] == "bandstop"):
        params["fc"] = [params["fc1"], params["fc2"]]
        del params["fc1"]
        del params["fc2"]
    else:
        params["fc"] = params["fc1"]
        del params["fc1"]
        del params["fc2"]
    output = freq_dom_(data).dig_filt(**params)
    output = output.tolist()
    return jsonify(output)


@app.route("/freq_dom_/wie_filt", methods=["POST"])
def wie_filt():
    data = request.get_json()["data"]
    params = {}
    for key in params.keys():
        try:
            value_type = type(params[key])
            params[key] = value_type(request.get_json()[key])
        except:
            pass
    output = freq_dom_(data).wie_filt(**params)
    output = output.tolist()
    return jsonify(output)


@app.route("/freq_dom_/kal_filt", methods=["POST"])
def kal_filt():
    data = request.get_json()["data"]
    params = {}
    for key in params.keys():
        try:
            value_type = type(params[key])
            params[key] = value_type(request.get_json()[key])
        except:
            pass
    output = freq_dom_(data).kal_filt(**params)
    output = output.tolist()
    return jsonify(output)


@app.route("/freq_dom_/lms_filt", methods=["POST"])
def lms_filt():
    data = request.get_json()["data"]
    params = {
        "dn": [0.0],
        "M": 8,
        "mu": 1e-3,
    }
    for key in params.keys():
        try:
            if key == "dn":
                params[key] = request.get_json()[key]
                continue
            value_type = type(params[key])
            params[key] = value_type(request.get_json()[key])
        except:
            pass
    temp = params["dn"]
    params["dn"] = [float(x) for x in temp.split(",")]
    output = freq_dom_(data).lms_filt(**params)
    return jsonify(output)


@app.route("/freq_dom_/get_harm", methods=["POST"])
def get_harm():
    data = request.get_json()["data"]
    params = {
        "fs": 256,
        "cut": 0.3,
        "flag_xd2pi": 0,
    }
    for key in params.keys():
        try:
            value_type = type(params[key])
            params[key] = value_type(request.get_json()[key])
        except:
            pass
    output = freq_dom_(data).get_harm(**params)
    return jsonify(output)


@app.route("/freq_dom_/denoise", methods=["POST"])
def denoise():
    data = request.get_json()["data"]
    params = {
        "method": "svd",
        "ratio": 0.01,
        "kernel_size": 5,
    }
    for key in params.keys():
        try:
            value_type = type(params[key])
            params[key] = value_type(request.get_json()[key])
        except:
            pass
    output = freq_dom_(data).denoise(**params)
    output = output.tolist()
    return jsonify(output)


@app.route("/freq_dom_/get_psd", methods=["POST"])
def get_psd():
    data = request.get_json()["data"]
    params = {
        "fs": 256,
        "method": "periodogram",
        "window": "hann",
        "nperseg": 256,
        "noverlap": 128,
        "flag_xd2pi": 0,
    }
    for key in params.keys():
        try:
            value_type = type(params[key])
            params[key] = value_type(request.get_json()[key])
        except:
            pass
    output = freq_dom_(data).get_psd(**params)
    return jsonify(output)


@app.route("/time_freq_dom_/get_wavpack", methods=["POST"])
def get_wavpack():
    data = request.get_json()["data"]
    params = {
        "wavelet": "db4",
        "level": 3,
        "mode": "symmetric",
        "order": "freq",
    }
    for key in params.keys():
        try:
            value_type = type(params[key])
            params[key] = value_type(request.get_json()[key])
        except:
            pass
    output = time_freq_dom_(data).get_wavpack(**params)
    return jsonify(output)


@app.route("/time_freq_dom_/get_ewt", methods=["POST"])
def get_ewt():
    data = request.get_json()["data"]
    params = {
        "N": 5,
        "log": 0,
        "detect": "locmax",
        "completion": 0,
        "reg": "none",
        "lengthFilter": 10,
        "sigmaFilter": 1,
    }
    for key in params.keys():
        try:
            value_type = type(params[key])
            params[key] = value_type(request.get_json()[key])
        except:
            pass
    output = time_freq_dom_(data).get_ewt(**params)
    return jsonify(output)


@app.route("/time_freq_dom_/get_lwt", methods=["POST"])
def get_lwt():
    data = request.get_json()["data"]
    params = {}
    for key in params.keys():
        try:
            value_type = type(params[key])
            params[key] = value_type(request.get_json()[key])
        except:
            pass
    output = time_freq_dom_(data).get_lwt(**params)
    return jsonify(output)


@app.route("/time_freq_dom_/get_emd", methods=["POST"])
def get_emd():
    data = request.get_json()["data"]
    params = {}
    for key in params.keys():
        try:
            value_type = type(params[key])
            params[key] = value_type(request.get_json()[key])
        except:
            pass
    output = time_freq_dom_(data).get_emd(**params)
    output = output.tolist()
    return jsonify(output)


@app.route("/time_freq_dom_/get_hht", methods=["POST"])
def get_hht():
    data = request.get_json()["data"]
    params = {
        "fs": 256,
        "intervals": 100,
    }
    for key in params.keys():
        try:
            value_type = type(params[key])
            params[key] = value_type(request.get_json()[key])
        except:
            pass
    output = time_freq_dom_(data).get_hht(**params)
    return jsonify(output)


@app.route("/time_freq_dom_/get_stfs", methods=["POST"])
def get_stfs():
    data = request.get_json()["data"]
    params = {
        "fs": 256,
        "window": "hann",
        "nperseg": 256,
        "noverlap": 64,
    }
    for key in params.keys():
        try:
            value_type = type(params[key])
            params[key] = value_type(request.get_json()[key])
        except:
            pass
    output = time_freq_dom_(data).get_stfs(**params)
    return jsonify(output)


@app.route("/time_freq_dom_/get_rnd", methods=["POST"])
def get_rnd():
    data = request.get_json()["data"]
    params = {
        "num": 1,
    }
    for key in params.keys():
        try:
            value_type = type(params[key])
            params[key] = value_type(request.get_json()[key])
        except:
            pass
    output = time_freq_dom_(data).get_rnd(**params)
    return jsonify(output)


@app.route("/tsf/lstm", methods=["POST"])
def test_lstm():
    params = {
        "seq_len": 50,
        "train_ratio": 0.8,
    }
    for key in params.keys():
        try:
            value_type = type(params[key])
            params[key] = value_type(request.get_json()[key])
        except:
            pass

    _, val_loader = TSF_Data(seq_len=params["seq_len"]).get_loader(
        train_ratio=params["train_ratio"]
    )

    pred, tgt, RMSE = TST_LSTM().get_ans(val_loader)
    output = [pred, tgt, RMSE]
    return jsonify(output)


@app.route("/tsf/elm", methods=["POST"])
def test_elm():
    params = {
        "seq_len": 50,
        "L": 30,
        "train_ratio": 0.8,
    }
    for key in params.keys():
        try:
            value_type = type(params[key])
            params[key] = value_type(request.get_json()[key])
        except:
            pass

    x_pool, y_pool = TSF_Data(seq_len=params["seq_len"]).get_pool()
    elm = TSF_ELM(
        L=params["L"],
        seq_len=params["seq_len"],
    )
    pred, tgt, RMSE = elm.get_ans(x_pool, y_pool, params["train_ratio"])
    output = [pred, tgt, RMSE]
    return jsonify(output)


@app.route("/tsf/svr", methods=["POST"])
def test_svr():
    params = {
        "seq_len": 50,
        "gamma": 0.001,
        "C": 10,
        "train_ratio": 0.8,
    }
    for key in params.keys():
        try:
            value_type = type(params[key])
            params[key] = value_type(request.get_json()[key])
        except:
            pass

    x_pool, y_pool = TSF_Data(seq_len=params["seq_len"]).get_pool()
    svr = TSF_SVR(gamma=params["gamma"], C=params["C"])
    pred, tgt, RMSE = svr.get_ans(x_pool, y_pool, params["train_ratio"])
    output = [pred, tgt, RMSE]
    return jsonify(output)


if __name__ == "__main__":
    app.run(host="0.0.0.0")
