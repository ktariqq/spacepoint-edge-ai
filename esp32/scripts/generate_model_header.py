"""
Turns the .tflite file into a C header the Arduino sketch can #include.
"""

with open("esp32/models/land_cover_model.tflite", "rb") as f:
    data = f.read()

with open("esp32/models/model_data.h", "w") as f:
    f.write("#ifndef MODEL_DATA_H\n#define MODEL_DATA_H\n\n")
    f.write(f"const unsigned int model_data_len = {len(data)};\n")
    f.write("alignas(8) const unsigned char model_data[] = {\n")
    for i in range(0, len(data), 12):
        chunk = data[i:i+12]
        f.write("  " + ", ".join(f"0x{b:02x}" for b in chunk) + ",\n")
    f.write("};\n\n#endif\n")

print("Wrote models/model_data.h")