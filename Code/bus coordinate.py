import pandas as pd
import twd97

file_path = r"C:\Users\user\Downloads\all.csv"

# Coba berbagai encoding
encodings = ['big5', 'cp950', 'utf-8', 'latin-1', 'big5hkscs']

print("=" * 65)
print("MEMBACA FILE CSV DENGAN BERBAGAI ENCODING")
print("=" * 65)

for enc in encodings:
    try:
        df = pd.read_csv(file_path, encoding=enc, nrows=3)
        print(f"\n✅ BERHASIL dengan encoding: {enc}")
        print(f"   Jumlah kolom: {len(df.columns)}")
        print(f"   Nama kolom: {df.columns.tolist()[:5]}...")
        
        # Tampilkan sample data
        print("\n   Sample data (baris pertama):")
        for col in df.columns[:5]:
            val = df[col].iloc[0]
            print(f"      {col}: {str(val)[:50]}")
        
        # Simpan encoding yang berhasil
        successful_encoding = enc
        break
    except Exception as e:
        print(f"❌ Gagal dengan {enc}: {str(e)[:50]}")

# Baca full file dengan encoding yang berhasil
print("\n" + "=" * 65)
print("MEMBACA FULL FILE")
print("=" * 65)

df_full = pd.read_csv(file_path, encoding=successful_encoding)
print(f"Total baris: {len(df_full)}")
print(f"Total kolom: {len(df_full.columns)}")
print(f"\nNama kolom lengkap:")
for i, col in enumerate(df_full.columns):
    print(f"   {i}: {col}")

# Cari kolom koordinat
print("\n" + "=" * 65)
print("MENCARI KOLOM KOORDINAT")
print("=" * 65)

x_col = None
y_col = None

for col in df_full.columns:
    col_lower = str(col).lower()
    if 'x' in col_lower or 'x坐' in col or 'twd97' in col_lower:
        # Cek apakah nilainya angka
        sample = df_full[col].dropna().iloc[0] if len(df_full[col].dropna()) > 0 else None
        if sample and str(sample).replace('.', '').replace('-', '').isdigit():
            x_col = col
            print(f"  Kolom X: {col} (sample: {sample})")
    if 'y' in col_lower or 'y坐' in col:
        sample = df_full[col].dropna().iloc[0] if len(df_full[col].dropna()) > 0 else None
        if sample and str(sample).replace('.', '').replace('-', '').isdigit():
            y_col = col
            print(f"  Kolom Y: {col} (sample: {sample})")

# Konversi jika ditemukan
if x_col and y_col:
    print("\n" + "=" * 65)
    print("KONVERSI KE WGS84")
    print("=" * 65)
    
    # Ambil sample untuk konversi
    sample_df = df_full[[x_col, y_col]].dropna().head(10)
    
    for idx, row in sample_df.iterrows():
        try:
            x = float(row[x_col])
            y = float(row[y_col])
            lng, lat = twd97.towgs84(x, y)
            print(f"  TWD97({x:.2f}, {y:.2f}) -> WGS84({lng:.6f}, {lat:.6f})")
        except Exception as e:
            print(f"  Error konversi: {e}")
    
    # Konversi semua data
    print("\nMelakukan konversi untuk semua data...")
    df_full['longitude'] = None
    df_full['latitude'] = None
    
    for idx, row in df_full.iterrows():
        try:
            x = float(row[x_col])
            y = float(row[y_col])
            lng, lat = twd97.towgs84(x, y)
            df_full.at[idx, 'longitude'] = lng
            df_full.at[idx, 'latitude'] = lat
        except:
            pass
    
    # Simpan hasil
    output_file = r"C:\Users\user\Downloads\taipower_coordinates_wgs84.csv"
    df_full.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n Hasil konversi disimpan ke: {output_file}")
    
    # Statistik
    n_success = df_full['longitude'].notna().sum()
    print(f"   Berhasil dikonversi: {n_success}/{len(df_full)} ({100*n_success/len(df_full):.1f}%)")
    
else:
    print("\n Tidak ditemukan kolom koordinat!")
    print("   Silakan periksa nama kolom di atas dan beri tahu saya.")

print("\n" + "=" * 65)