import numpy as np

#%% 
UQ_NIDS_LABEL_DICT = {"DoS": ["dos", "DoS attacks-Hulk", "DoS attacks-SlowHTTPTest",
                              "DoS attacks-GoldenEye", "DoS attacks-Slowloris", "DoS"],
                      "DDoS": ["ddos", "DDoS attacks-LOIC-HTTP", "DDoS", "DDOS attack-LOIC-UDP",
                               "DDOS attack-HOIC"],
                      "Brute Force": ["Brute Force -Web", "Brute Force -XSS", "FTP-BruteForce",
                                      "SSH-Bruteforce"],
                      "injection": ["SQL Injection", "injection"],
                      "Backdoor": ["Backdoor", "backdoor"]
                      }

WEBSITE_LABEL_DICT = {"DoS": ["dos", "DoS attacks-Hulk", "DoS attacks-SlowHTTPTest",
                              "DoS attacks-GoldenEye", "DoS attacks-Slowloris", "DoS"],
                      "DDoS": ["ddos", "DDoS attacks-LOIC-HTTP", "DDoS", "DDOS attack-LOIC-UDP",
                               "DDOS attack-HOIC"],
                      "Brute Force": ["Brute Force -Web", "Brute Force -XSS", "FTP-BruteForce",
                                      "SSH-Bruteforce"],
                      "Injection": ["SQL Injection", "injection"],
                      "Backdoor": ["Backdoor", "backdoor"],
                      "MITM": ["mitm"],"XSS": ["xss"], "Scanning": ["scanning"],
                      "Ransomware": ["ransomware"], "Password": ["password"]
                      }

ZEEK_LABEL_DICT = {"DoS": ["DoS - Hulk", "DoS - SlowHTTPTest", "DoS - GoldenEye", "DoS - Slowloris", "DoS"],
                   "DDoS": ["DDoS - HOIC", "DDoS - LOIC - HTTP", "DDoS - LOIC", "DDoS - LOIC - UDP"], 
                   "Bot": ["DDoS - Botnet"],
                   "Brute Force": ["Web Attack - Brute Force", "SSH Brute Force", "Patator - FTP", "Patator - SSH"],
                   "Infiltration": ["Infiltration - Cool Disk", "Infiltration - Dropbox Download", "Infiltration"]
                   }

#%% 
def group_labels(data, select_dict = None):
    if select_dict == None:
        raise ValueError("Select dict cannot be None")
    if select_dict == "UQ-NIDS":
        conversion_dict = UQ_NIDS_LABEL_DICT
    if select_dict == "UQ-NIDS website":
        conversion_dict = WEBSITE_LABEL_DICT
    if select_dict == "Zeek":
        conversion_dict = ZEEK_LABEL_DICT
    
    for key, value in conversion_dict.items():
        data["Label"] = np.where(data["Label"].isin(value), key, data["Label"])
    return data
        
