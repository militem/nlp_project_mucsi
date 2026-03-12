import pandas as pd
import json
import os

def convert_json_to_csv(json_path, csv_path):
    print(f"Loading data from {json_path}...")
    if not os.path.exists(json_path):
        print(f"Error: {json_path} not found.")
        return

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    df = pd.DataFrame(data)
    initial_count = len(df)
    print(f"Initial rows: {initial_count}")

    # Identificamos los duplicados antes de eliminarlos
    subset_cols = ['name', 'address', 'price', 'rating']
    duplicates = df[df.duplicated(subset=subset_cols, keep=False)]
    
    if not duplicates.empty:
        print("\n--- Filas duplicadas encontradas ---")
        # Agrupamos por los campos clave para mostrar los grupos de duplicados
        for _, group in duplicates.groupby(subset_cols):
            print(group[['name', 'address', 'price', 'rating', 'url']].to_string())
            print("-" * 50)
    else:
        print("\nNo se encontraron filas duplicadas con los criterios: ", subset_cols)

    # Remove duplicates based on name, address, price, and rating (user's updated criteria)
    df_clean = df.drop_duplicates(subset=subset_cols, keep='first')
    
    final_count = len(df_clean)
    print(f"Rows after removing duplicates: {final_count}")
    print(f"Removed {initial_count - final_count} duplicates.")

    # Export to CSV
    # Using index=False to avoid adding an extra column
    df_clean.to_csv(csv_path, index=False, encoding='utf-8')
    print(f"Saved cleaned data to {csv_path}")

if __name__ == "__main__":
    base_path = "/home/alex/Documents/master/mineria_texto_nlp/proyecto"
    json_file = os.path.join(base_path, "expedia_hotels.json")
    csv_file = os.path.join(base_path, "expedia_hotels.csv")
    
    convert_json_to_csv(json_file, csv_file)
