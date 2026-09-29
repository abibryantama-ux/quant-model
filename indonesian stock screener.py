# ==============================================================================
# COPYRIGHT & INTELLECTUAL PROPERTY NOTICE
# Copyright (c) 2026 Noorman Abi Bryantama. All rights reserved.
# 
# NOTICE: This is a sanitized portfolio version. The core proprietary 
# mathematical formulas and custom weights inside the Merton Jump Diffusion 
# simulation block have been masked/abstracted to protect intellectual property. 
# Code architecture, data pipelines, and filtering layers remain fully visible.
# ==============================================================================

import os
import time
import numpy as np
import pandas as pd
import yfinance as yf

# ==============================================================================
# CONFIGURATION TARGET CORE MAKRO (PASAR INDONESIA - 7 SENSOR INSTITUSIONAL)
# ==============================================================================
JUMLAH_SIMULASI = 10000
HARI_KE_DEPAN = 1
THRESHOLD_PELUANG = 0.10       
FILE_DAFTAR_EMITEN = "daftar_emiten_indo.txt" 
FILE_OUTPUT_HASIL = "hasil_analisis_merton_indo.xlsx"  

# AMBANG BATAS SENSOR BURSA EFEK INDONESIA (IDX)
MINIMAL_NILAI_TRANSAKSI_IDR = 10000000000  
BATAS_MAKSIMAL_PE = 35                     
BATAS_MAKSIMAL_DER = 200                   
MINIMAL_HARGA_NOMINAL = 100.0              
# ==============================================================================

print("=" * 135)
print("📱 IDX MULTI-ASSET MERTON PIPELINE - PEMINDAIAN 7 LEVEL SENSOR FUNDAMENTAL & GROWTH REAL-TIME")
print("=" * 135)

if not os.path.exists(FILE_DAFTAR_EMITEN):
    print(f"❌ Error: File '{FILE_DAFTAR_EMITEN}' tidak ditemukan!")
    exit()

with open(FILE_DAFTAR_EMITEN, "r") as f:
    list_emiten = [line.strip().upper() for line in f if line.strip()]

total_emiten = len(list_emiten)
print(f"✅ Berhasil memuat {total_emiten} emiten dari data offline IDX.")
print("Memulai pemindaian tingkat sensor mendalam harian...\n")

semua_hasil = []

# 2. PROSES PIPELINE LOOPING SAHAM INDONESIA
for index, kode in enumerate(list_emiten, 1):
    print(f"[{index}/{total_emiten}] Ticker: {kode} ... ", end="", flush=True)
    
    if not kode.endswith(".JK"):
        print("❌ SKIP (Bukan .JK)")
        continue
    
    try:
        saham = yf.Ticker(kode)
        df_historis = saham.history(period="1y")

        if len(df_historis) < 200:
            print("❌ SKIP (Data < 200 hari)")
            continue

        harga_pasar_saat_ini = float(df_historis['Close'].iloc[-1])
        volume_terakhir = int(df_historis['Volume'].iloc[-1])

        # SENSOR A: LIKUIDITAS & TEKNIKAL
        if harga_pasar_saat_ini < MINIMAL_HARGA_NOMINAL:
            print(f"❌ SKIP (Harga < Rp{MINIMAL_HARGA_NOMINAL:,.0f})")
            continue

        df_historis['Nilai_Transaksi'] = df_historis['Volume'] * df_historis['Close']
        rata_rata_likuiditas = df_historis['Nilai_Transaksi'].tail(20).mean()
        rata_rata_miliar = rata_rata_likuiditas / 1e9 
        
        if rata_rata_likuiditas < MINIMAL_NILAI_TRANSAKSI_IDR:
            print(f"❌ SKIP (Sepi, Rp{rata_rata_miliar:.1f} M/hari)")
            continue

        ma_200 = df_historis['Close'].rolling(window=200).mean().iloc[-1]
        if harga_pasar_saat_ini < ma_200:
            print("❌ SKIP (Bearish, < MA 200)")
            continue

        # SENSOR B: FUNDAMENTAL RATIO & GROWTH EXTRACTION
        info_saham = saham.info
        roe = info_saham.get("returnOnEquity")
        pe_ratio = info_saham.get("trailingPE")
        debt_to_equity = info_saham.get("debtToEquity")
        eps_growth = info_saham.get("earningsGrowth")

        if roe is None or roe <= 0:
            print("❌ SKIP (ROE Negatif/Rugi)")
            continue

        if pe_ratio is None or pe_ratio <= 0 or pe_ratio > BATAS_MAKSIMAL_PE:
            print(f"❌ SKIP (P/E tidak ideal / > {BATAS_MAKSIMAL_PE})")
            continue

        if debt_to_equity is not None and debt_to_equity > BATAS_MAKSIMAL_DER:
            print(f"❌ SKIP (Utang Tinggi, DER: {debt_to_equity:.1f}%)")
            continue
            
        if eps_growth is not None and eps_growth <= 0:
            print(f"❌ SKIP (Laba Mengalami Penurunan / EPS Growth: {eps_growth*100:.1f}%)")
            continue

        # PROSES INTI QUANTITATIVE MODELING
        df = df_historis.tail(90).copy().sort_index()
        df['Log_Return'] = np.log(df['Close'] / df["Close"].shift(1))
        log_returns = df['Log_Return'].dropna()
        
        volatilitas_total = log_returns.std(ddof=1)
        drift_aktual = log_returns.mean()
        batas_shock_historis = volatilitas_total * harga_pasar_saat_ini
        
        hasil_dua_zona = {}

        for jenis_uji, harga_uji in [("SUPPORT_SHOCK", harga_pasar_saat_ini - batas_shock_historis), 
                                     ("RESISTANCE_SHOCK", harga_pasar_saat_ini + batas_shock_historis)]:

            status_posisi = "BAWAH" if harga_uji < harga_pasar_saat_ini else "ATAS"

            # ------------------------------------------------------------------
            # PROPRIETARY PROTECTION ZONING (RUMUS INTI DI-MASKING)
            # ------------------------------------------------------------------
            # Catatan Portofolio: Bagian ini menggunakan simulasi stokastik standar
            # untuk mendemonstrasikan kapabilitas structural coding Python tanpa
            # membocorkan koefisien Alpha & parameter Jump Diffusion komersial.
            
            W = np.random.normal(0, 1, JUMLAH_SIMULASI)
            komp_drift = drift_aktual * HARI_KE_DEPAN
            komp_difusi = volatilitas_total * np.sqrt(HARI_KE_DEPAN) * W
            efek_jump = np.random.normal(0, 0.01, JUMLAH_SIMULASI) # Placeholder noise
            
            prediksi_harga_pasar = harga_pasar_saat_ini * np.exp(komp_drift + komp_difusi + efek_jump)
            jarak_deviasi = abs(harga_pasar_saat_ini - harga_uji)
            # ------------------------------------------------------------------

            if status_posisi == "BAWAH":
                target_atas_cermin = harga_pasar_saat_ini + jarak_deviasi
                peluang_naik = np.mean(prediksi_harga_pasar > target_atas_cermin)
                peluang_turun = np.mean(prediksi_harga_pasar < harga_uji)
            else:
                target_bawah_cermin = harga_pasar_saat_ini - jarak_deviasi
                peluang_naik = np.mean(prediksi_harga_pasar > harga_uji)
                peluang_turun = np.mean(prediksi_harga_pasar < target_bawah_cermin)

            rekomendasi = "WAIT (Belum Jenuh)"
            if status_posisi == "BAWAH" and peluang_turun < THRESHOLD_PELUANG:
                rekomendasi = "🔥 BIDIK BUY! (Jenuh Jual)"
            elif status_posisi == "ATAS" and peluang_naik < THRESHOLD_PELUANG:
                rekomendasi = "⚠️ HATI-HATI SELL! (Jenuh Beli)"

            hasil_dua_zona[jenis_uji] = {
                "Ticker": kode,
                "Harga_Live": harga_pasar_saat_ini,
                "Transaksi_Avg_Miliar": round(rata_rata_miliar, 2),
                "ROE_Pct": round(roe * 100, 2) if roe is not None else "N/A",
                "P/E_Ratio": round(pe_ratio, 2) if pe_ratio is not None else "N/A",
                "DER_Pct": round(debt_to_equity, 2) if debt_to_equity is not None else "N/A",
                "EPS_Growth%": round(eps_growth * 100, 2) if eps_growth is not None else "N/A",
                "Tipe_Zona": jenis_uji,
                "Harga_Uji": round(harga_uji, 2),
                "P_Naik%": round(peluang_naik * 100, 1),
                "P_Turun%": round(peluang_turun * 100, 1),
                "Sinyal_Akhir": rekomendasi
            }

        s_bawah = hasil_dua_zona["SUPPORT_SHOCK"]["Sinyal_Akhir"]
        s_atas = hasil_dua_zona["RESISTANCE_SHOCK"]["Sinyal_Akhir"]

        if "BUY" in s_bawah and "SELL" in s_atas:
            hasil_dua_zona["SUPPORT_SHOCK"]["Sinyal_Akhir"] = "🔄 KONSOLIDASI KETAT"
            hasil_dua_zona["RESISTANCE_SHOCK"]["Sinyal_Akhir"] = "🔄 KONSOLIDASI KETAT"

        semua_hasil.append(hasil_dua_zona["SUPPORT_SHOCK"])
        semua_hasil.append(hasil_dua_zona["RESISTANCE_SHOCK"])
            
        print(f"✅ OK (Avg: {rata_rata_miliar:.1f} M/hari)")
        
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")

# 3. EXPORT KE EXCEL DATA FRAME HASIL PEMINDAIAN MODEL
print("\n" + "=" * 135)
if semua_hasil:
    df_final = pd.DataFrame(semua_hasil)
    df_final.to_excel(FILE_OUTPUT_HASIL, index=False)
    print(f"📊 Pemindaian selesai! Berhasil mengekspor {len(df_final)} baris analisis ke '{FILE_OUTPUT_HASIL}'.")
else:
    print("⚠️ Tidak ada emiten yang lolos sensor pemindaian hari ini.")
print("=" * 135)
