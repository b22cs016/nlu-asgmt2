def process_names(input_file, output_file):
    try:
        # 1. Read the raw text from data.txt
        with open(input_file, 'r', encoding='utf-8') as f:
            raw_text = f.read()

        # 2. Split the text by commas
        raw_names_list = raw_text.split(',')

        # 3. Clean up the names (remove spaces and ignore empty strings)
        cleaned_names = []
        for name in raw_names_list:
            clean_name = name.strip() # Removes leading/trailing spaces or newlines
            if clean_name:            # Only add if it's not empty
                cleaned_names.append(clean_name)

        # 4. Write them line-by-line to TrainingNames.txt
        with open(output_file, 'w', encoding='utf-8') as f:
            for name in cleaned_names:
                f.write(name + '\n')

        print(f"Success! Saved {len(cleaned_names)} clean names to {output_file}")

    except FileNotFoundError:
        print(f"Error: Could not find {input_file}. Make sure it is in the same folder.")

# Run the function
if __name__ == "__main__":
    process_names('data.txt', 'TrainingNames.txt')