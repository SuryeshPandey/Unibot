     print(f"🟨 Incoming Query: {user_input}")

        # Clean possessives: "Mishra's" → "Mishra"
        cleaned_input = re.sub(r"'s", "", user_input)

        # Clean irrelevant keywords
        cleaned_input = re.sub(r"\b(cabin|room|office|location|of|the)\b", "", cleaned_input, flags=re.IGNORECASE)

        # Actual name extraction (expecting 2 capitalized words)
        name_match = re.search(r"(?:professor|dr\.?|mr\.?|ms\.?)?\s*([A-Z][a-z]+ [A-Z][a-z]+)", cleaned_input)

        if name_match:
            prof_name = name_match.group(1).strip()
            print(f"🧠 Looking up cabin for: {prof_name}")

            conn = sqlite3.connect(DB_NAME)
            cursor = conn.cursor()
            cursor.execute("SELECT name, cabin FROM professors WHERE name LIKE ?", (f"%{prof_name}%",))
            row = cursor.fetchone()
            conn.close()

            if row:
                name, cabin = row
                return jsonify({"response": f"{name} sits in cabin {cabin}."})
            else:
                return jsonify({"response": f"I couldn't find {prof_name}'s cabin in the database."})
        else:
            print("❌ Regex failed to find professor name.")
            return jsonify({"response": "Please mention the professor's name clearly."})