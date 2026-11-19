- Download dataset from project instructions
- Unpack dataset so file structure is:

   `data/stage_4_data/text_classification` and `data/stage_4_data/text_generation`
- Run `python movie_clean_dataset.py`
  - optionally: Run `python movie_validate_clean_csv.py`
- Run `python movie_build_vocab.py`
  - optionally: Run `python movie_validate_encoded.py`
- Run `python movie_test_dataset_loader.py`