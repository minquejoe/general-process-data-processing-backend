# general-process-data-processing-backend
A backend intergrated some general open sources process data processing methods, including statistics, time domain, frequency domain,  time-freq domian and time series forecasting(only for test).

# usage
1. run file：`python np-4.2_test_api.py`

1. send http request：`curl --request POST --url http://[address]/[alg_class]/[alg] --header 'content-type: application/json' --header 'user-agent: vscode-restclient' --data '{"data": [data], ["optional param 1": param values 1], ["optional param 2": param values 2], ……}'`

optional params see `np_4_2.py` and `np_4_2_tsf.py`, the original method file, or the params dict in every route in the file `np-4.2_test_api.py`.

# example
`curl --request POST --url http://127.0.0.1:5000/freq_dom_/denoise --header 'content-type: application/json' --header 'user-agent: vscode-restclient' --data '{"data": [1,2,3], "ratio": 0.1}'`

**explain**: 
backend run on 127.0.0.1:5000，use denoise methond in  frequency domain, default use svd method, the top 1% singular values are remained.

**result**：
HTTP/1.1 200 OK
Server: Werkzeug/2.2.3 Python/3.8.8
Date: Sun, 07 May 2023 10:33:05 GMT
Content-Type: application/json
Content-Length: 14
Connection: close

[
  0.0,
  0.0,
  0.0
]
