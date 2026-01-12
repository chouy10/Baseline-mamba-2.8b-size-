# Baseline-mamba-2.8b-size-



The design passes synthesis, but the BRAM utilization is too high and still needs optimization:
![newsize syn](https://github.com/user-attachments/assets/8953b8c7-0a9e-40dd-8fe8-b342bd1bc514)
%
![newsize syn %](https://github.com/user-attachments/assets/ec576775-f9a2-4a7d-86f9-fe8384c340d5)

The design passes co-simulation, and the latency is shown below.
![newsize cosim](https://github.com/user-attachments/assets/cb29c37c-8b64-4e81-aa82-225d19b6c27f)

There is still an issue with excessive depth causing overly high BRAM utilization that needs to be resolved.


