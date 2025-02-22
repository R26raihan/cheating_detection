![example](https://github.com/user-attachments/assets/99362543-589f-439a-ad5d-78a0b9fe18c8)


# Cara Perhitungan

Turn Count = 10 (peserta menoleh 10X)
Nod Count = 5 (peserta menunduk 5x)
Items count = 2 (terdeteksi Membawa Barang Mencurigakan)
calculator count 1 (terdeteksi membawa hp atau kalkulator) 

# Bobot
- w1 = 1.0(untuk menoleh)
- w2 = 1.5 (untuk menunduk)
- w3 = 2.0 (untuk barang mencurigakan)
- w4 = 2.5 ( untuk kalkulator atau HP)

Total Possible Actions = 1200 (untuk ujian 2 jam)

# Hitung Total Suspicious Actions
- Total Suspicious Actions= (10 x 1.0 ) + (5 x 1.5) + (2 x 2.0) + (1 x 2.5)
- Total Suspicious Actions= 10 + 7.5 + 4 + 2.5 = 24

#Hitung Persentase Kecurangan#
Cheating Percentage = (24/1200) X 100 = 2%


# requirements
- Flask==2.3.2
- opencv-python==4.8.0.76
- mediapipe==0.10.0
- numpy==1.24.3
- ultralytics==8.0.124
- flask-socketio==5.3.4



