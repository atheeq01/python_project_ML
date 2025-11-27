# Cosmetic Product Database Guide

This folder contains a CSV file (`cosmetic_products_db.csv`) designed for use with RAG (Retrieval-Augmented Generation) chatbots.

## How to use this with your Chatbot

**Upload:** Upload the `.csv` file to your chatbot's "Knowledge Base" or "Documents" section.

**Indexing:** Ensure your system indexes the file. Most systems (like OpenAI GPTs, LangChain, or Flowise) handle CSV headers automatically.

## CSV Column Dictionary

When instructing your chatbot, you can refer to these columns:

- **product_id:** Unique identifier (useful for inventory lookups).
- **product_name:** The commercial name of the item.
- **brand:** (Cetaphil, The Ordinary, CeraVe).
- **category:** Broad grouping (Cleanser, Serum, Moisturizer, Sunscreen).
- **skin_type:** Who should use this (e.g., "Oily", "Dry", "Sensitive").
- **concern:** Specific problems the product solves (e.g., "Acne", "Aging", "Dullness").
- **description:** The "Hook" text. Used by the AI to sell the product.
- **key_ingredients:** Active chemicals (helpful for "Does this have Retinol?" questions).
- **how_to_use:** Step-by-step instructions.
- **pro_tip:** Expert advice to make the chatbot sound smarter.

## Example Chatbot System Prompt

If you are configuring the "System Message" for your bot, use this structure to make the best use of this CSV:

```
You are an expert beauty consultant for a cosmetic shop. You have access to a database of products.

When a user asks for a recommendation:
- Search the 'skin_type' and 'concern' columns to find matches.
- Present the 'product_name' and the 'description'.
- Always end with the 'pro_tip' for that product to add value.

If a user asks how to use a product, strictly follow the 'how_to_use' column steps.
```

## Notes on the Data

- **Cetaphil:** Covers the core cleansers, moisturizers, and the Bright Healthy Radiance line.
- **The Ordinary:** Includes the best-sellers (Niacinamide, Retinol, Glycolic, etc.). Some generic terms from your list were consolidated into specific popular products to ensure accuracy.
- **CeraVe:** Consolidated duplicates (e.g., "Moisturising Lotion" vs "CeraVe Moisturising Lotion") into single, robust entries.