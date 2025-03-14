dwt_coding lab guidance: 

sudo apt update
sudo apt install git

git init
git pull https://github.com/vuongnguyen168/lab_1_dwt_code.git main --allow-unrelated-histories(Unpacking objects)

pip install PyWavelets opencv-python Pillow

python3 dwt_encrypt.py (secret message)

ls(encrypted.png)

python3 dwt_decrypt.py >> plaintext.txt(secret message)

cat plaintext.txt


