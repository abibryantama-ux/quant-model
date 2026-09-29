import os, time
import numpy as np
import pandas as pd
import yfinance as yf

# ==============================================================================
# KONFIGURASI UTAMA
# ==============================================================================
JUMLAH_SIMULASI = 10000
HARI_KE_DEPAN = 1
THRESHOLD_PANTAUAN = 0.10      # Batas Pantau <10%
THRESHOLD_EKSEKUSI = 0.05      # Batas Eksekusi <5%
FILE_DAFTAR_EMITEN = "daftar_emiten_us.txt"
FILE_OUTPUT_HASIL = "hasil_analisis_merton_us.xlsx"  

# AMBANG BATAS PENYARINGAN
MINIMAL_NILAI_TRANSAKSI_USD = 20000000     # Minimal 20 Juta USD/hari
BATAS_MAKSIMAL_PE = 45                     
BATAS_MAKSIMAL_DER = 150                   
MINIMAL_HARGA_NOMINAL = 10.0               
# ==============================================================================

print("=" * 135)
print("📱 ANALISIS MERTON US - VERSI RINGKAS | SATUAN USD")
print("=" * 135)

if not os.path.exists(FILE_DAFTAR_EMITEN):
    print(f"❌ File '{FILE_DAFTAR_EMITEN}' tidak ditemukan!")
    exit()

with open(FILE_DAFTAR_EMITEN, "r") as f:
    list_emiten = [line.strip().upper() for line in f if line.strip()]

total_emiten = len(list_emiten)
print(f"✅ Memuat {total_emiten} kode saham.\n")
semua_hasil = []

# PROSES PEMINDAIAN
for index, kode in enumerate(list_emiten, 1):
    print(f"[{index}/{total_emiten}] {kode} ... ", end="", flush=True)
    
    if kode.endswith(".JK"):
        print("❌ SKIP (Bukan Saham AS)")
        continue
    
    try:
        saham = yf.Ticker(kode)
        df_historis = saham.history(period="1y")

        if len(df_historis) < 90:
            print("❌ SKIP (Data < 90 hari)")
            continue

        harga_pasar_saat_ini = float(df_historis['Close'].iloc[-1])

        if harga_pasar_saat_ini < MINIMAL_HARGA_NOMINAL:
            print(f"❌ SKIP (Harga < ${MINIMAL_HARGA_NOMINAL})")
            continue

        # === PERHITUNGAN LIKUIDITAS ===
        df_historis['Nilai_Transaksi_USD'] = df_historis['Volume'] * df_historis['Close']
        vol_terakhir_lembar = int(df_historis['Volume'].iloc[-1])
        transaksi_terakhir_usd = float(df_historis['Nilai_Transaksi_USD'].iloc[-1])
        transaksi_terakhir_miliar = transaksi_terakhir_usd / 1e9
        rata_rata_likuiditas = df_historis['Nilai_Transaksi_USD'].tail(20).mean()
        rata_rata_miliar_usd = rata_rata_likuiditas / 1e9 
        
        if rata_rata_likuiditas < MINIMAL_NILAI_TRANSAKSI_USD:
            print("❌ SKIP (Kurang cair)")
            continue

        # === FILTER FUNDAMENTAL ===
        info_saham = saham.info
        roe = info_saham.get("returnOnEquity")
        pe_ratio = info_saham.get("trailingPE")
        debt_to_equity = info_saham.get("debtToEquity")
        eps_growth = info_saham.get("earningsGrowth")

        if roe is None or roe <= 0:
            print("❌ SKIP (ROE Negatif)")
            continue
        if pe_ratio is None or pe_ratio <= 0 or pe_ratio > BATAS_MAKSIMAL_PE:
            print(f"❌ SKIP (P/E > {BATAS_MAKSIMAL_PE})")
            continue
        if debt_to_equity is not None and debt_to_equity > BATAS_MAKSIMAL_DER:
            print("❌ SKIP (Utang Tinggi)")
            continue
        if eps_growth is not None and eps_growth <= 0:
            print("❌ SKIP (Laba Menurun)")
            continue

        # === MODEL MERTON ===
        df = df_historis.tail(90).copy().sort_index()
        df['Log_Return'] = np.log(df['Close'] / df["Close"].shift(1))
        log_returns = df['Log_Return'].dropna()
        
        volatilitas_total = log_returns.std(ddof=1)
        drift_aktual = log_returns.mean()
        batas_shock_historis = volatilitas_total * harga_pasar_saat_ini
        
        hasil_dua_zona = {}
        target_uji_list = [
            ("SUPPORT_SHOCK", harga_pasar_saat_ini - batas_shock_historis), 
            ("RESISTANCE_SHOCK", harga_pasar_saat_ini + batas_shock_historis)
        ]

        for jenis_uji, harga_uji in target_uji_list:
            status_posisi = "BAWAH" if harga_uji < harga_pasar_saat_ini else "ATAS"
            batas_jump = 1.5 * volatilitas_total
            jumps = log_returns[abs(log_returns) > batas_jump]
            
            lambda_jump = len(jumps) / len(log_returns) if len(log_returns) > 0 else 0.05
            mu_jump = jumps.mean() if len(jumps) > 0 else 0.0
            sigma_jump = jumps.std(ddof=1) if len(jumps) > 1 else 0.01
            volatilitas_dasar = np.sqrt(max(0.0001, volatilitas_total**2 - lambda_jump * (mu_jump**2 + sigma_jump**2)))

            W = np.random.normal(0, 1, JUMLAH_SIMULASI)
            N = np.random.poisson(lambda_jump * HARI_KE_DEPAN, JUMLAH_SIMULASI)
            max_jumps = int(np.max(N))
            efek_jump = np.zeros(JUMLAH_SIMULASI)
            if max_jumps > 0:
                for j in range(1, max_jumps + 1):
                    mask = (N >= j)
                    efek_jump[mask] += np.random.normal(mu_jump, sigma_jump, np.sum(mask))
            
            k = np.exp(mu_jump + 0.5 * sigma_jump**2) - 1
            komp_drift = (drift_aktual - lambda_jump * k) * HARI_KE_DEPAN
            komp_difusi = volatilitas_dasar * np.sqrt(HARI_KE_DEPAN) * W
            prediksi_harga_pasar = harga_pasar_saat_ini * np.exp(komp_drift + komp_difusi + efek_jump)
            jarak_deviasi = abs(harga_pasar_saat_ini - harga_uji)

            if status_posisi == "BAWAH":
                target_atas_cermin = harga_pasar_saat_ini + jarak_deviasi
                peluang_naik = np.mean(prediksi_harga_pasar > target_atas_cermin)
                peluang_turun = np.mean(prediksi_harga_pasar < harga_uji)
            else:
                target_bawah_cermin = harga_pasar_saat_ini - jarak_deviasi
                peluang_naik = np.mean(prediksi_harga_pasar > harga_uji)
                peluang_turun = np.mean(prediksi_harga_pasar < target_bawah_cermin)

            # REKOMENDASI
            rekomendasi = "WAIT (Belum Jenuh)"
            if status_posisi == "BAWAH":
                if peluang_turun < THRESHOLD_EKSEKUSI:
                    rekomendasi = "🔥 BIDIK BUY! (Jenuh Jual <5%)"
                elif peluang_turun < THRESHOLD_PANTAUAN:
                    rekomendasi = "WAIT (Peluang Beli <10%)"
            else:
                if peluang_naik < THRESHOLD_EKSEKUSI:
                    rekomendasi = "⚠️ HATI-HATI SELL! (Jenuh Beli <5%)"
                elif peluang_naik < THRESHOLD_PANTAUAN:
                    rekomendasi = "WAIT (Peluang Jual <10%)"

            hasil_dua_zona[jenis_uji] = {
                "Ticker": kode,
                "Harga_Terkini (USD)": round(harga_pasar_saat_ini, 2),
                "Volume_Terakhir (Lembar)": vol_terakhir_lembar,
                "Transaksi_Terakhir (Miliar USD)": round(transaksi_terakhir_miliar, 2),
                "Transaksi_Rata20 (Miliar USD)": round(rata_rata_miliar_usd, 2),
                "ROE (%)": round(roe * 100, 2) if roe is not None else "N/A",
                "P/E_Rasio": round(pe_ratio, 2) if pe_ratio is not None else "N/A",
                "DER (%)": round(debt_to_equity, 2) if debt_to_equity is not None else "N/A",
                "Pertumbuhan_Laba (%)": round(eps_growth * 100, 2) if eps_growth is not None else "N/A",
                "Tipe_Zona": jenis_uji,
                "Harga_Batas_Uji (USD)": round(harga_uji, 2),
                "Peluang_Naik (%)": round(peluang_naik * 100, 1),
                "Peluang_Turun (%)": round(peluang_turun * 100, 1),
                "Sinyal_Akhir": rekomendasi
            }

        # CEK KONSOLIDASI
        s_bawah = hasil_dua_zona["SUPPORT_SHOCK"]["Sinyal_Akhir"]
        s_atas = hasil_dua_zona["RESISTANCE_SHOCK"]["Sinyal_Akhir"]
        p_naik_val = hasil_dua_zona["RESISTANCE_SHOCK"]["Peluang_Naik (%)"]
        p_turun_val = hasil_dua_zona["SUPPORT_SHOCK"]["Peluang_Turun (%)"]
        selisih_peluang = abs(p_naik_val - p_turun_val)

        if ("BUY" in s_bawah or "Beli" in s_bawah) and ("SELL" in s_atas or "Jual" in s_atas):
            if selisih_peluang < 2.0:
                hasil_dua_zona["SUPPORT_SHOCK"]["Sinyal_Akhir"] = "🔄 KONSOLIDASI KETAT"
                hasil_dua_zona["RESISTANCE_SHOCK"]["Sinyal_Akhir"] = "🔄 KONSOLIDASI KETAT"

        semua_hasil.append(hasil_dua_zona["SUPPORT_SHOCK"])
        semua_hasil.append(hasil_dua_zona["RESISTANCE_SHOCK"])
            
        print(f"✅ OK | Rata2: ${rata_rata_miliar_usd:.1f} Miliar")
        time.sleep(0.7)

    except Exception as e:
        print(f"❌ ERROR: {str(e)[:50]}...")
        continue

# === SIMPAN HASIL ===
print("\n" + "=" * 135)
if semua_hasil:
    df_final = pd.DataFrame(semua_hasil)
    urutan_kolom = [
        "Ticker", "Harga_Terkini (USD)", "Volume_Terakhir (Lembar)",
        "Transaksi_Terakhir (Miliar USD)", "Transaksi_Rata20 (Miliar USD)",
        "ROE (%)", "P/E_Rasio", "DER (%)", "Pertumbuhan_Laba (%)",
        "Tipe_Zona", "Harga_Batas_Uji (USD)", "Peluang_Naik (%)", "Peluang_Turun (%)", "Sinyal_Akhir"
    ]
    df_final = df_final[urutan_kolom]
    df_final.to_excel(FILE_OUTPUT_HASIL, index=False)
    print(f"🎉 BERHASIL! Hasil tersimpan di: '{FILE_OUTPUT_HASIL}'")
else:
    print("❌ Tidak ada saham yang lolos syarat likuiditas & fundamental.")
print("=" * 135)
