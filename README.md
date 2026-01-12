# Baseline-mamba-2.8b-size-
The current Mamba configuration uses the same size as Mamba-2.8B:

<img width="470" height="250" alt="image" src="https://github.com/user-attachments/assets/f5fe64a5-04e5-4238-b131-7f68eec1065e" />

<img width="1159" height="223" alt="image" src="https://github.com/user-attachments/assets/aef637d3-c1a7-435f-b6fa-6f3a69a33cce" />


The design passes synthesis, but the BRAM utilization is too high and still needs optimization:
![newsize syn](https://github.com/user-attachments/assets/8953b8c7-0a9e-40dd-8fe8-b342bd1bc514)
%
![newsize syn %](https://github.com/user-attachments/assets/ec576775-f9a2-4a7d-86f9-fe8384c340d5)

The design passes co-simulation, and the latency is shown below:
![newsize cosim](https://github.com/user-attachments/assets/cb29c37c-8b64-4e81-aa82-225d19b6c27f)

There is still an issue with excessive depth causing overly high BRAM utilization that needs to be resolved.


