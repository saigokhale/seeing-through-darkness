# Seeing Through Darkness: Space Debris Detection
#### Creating a multimodal dataset and model by fusing thermal and visual image properties of space objects to detect and classify the objects into satellite and debris. Our fusion technique led to an improvement in accuracy by 30%, as compared to performance when trained solely on visible images.

### Objectives

* To create a solution that enables space debris detection in poor illumination conditions 
* Fusion of visible and thermal image fusion methodologies allows complementary image fusion to be combined, to ensure better parameterization
* Current methods only rely on SNR ratio, material color signature and attitudes using visible image classification techniques for debris detection


## Examples

* A sample of our fused dataset
* ![sample of dataset.](https://private-user-images.githubusercontent.com/71946733/288082964-3ea97c7e-5bcd-428d-9d41-1ef4e56b4423.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTEiLCJleHAiOjE3MDE3OTAxMTQsIm5iZiI6MTcwMTc4OTgxNCwicGF0aCI6Ii83MTk0NjczMy8yODgwODI5NjQtM2VhOTdjN2UtNWJjZC00MjhkLTlkNDEtMWVmNGU1NmI0NDIzLnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFJV05KWUFYNENTVkVINTNBJTJGMjAyMzEyMDUlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjMxMjA1VDE1MjMzNFomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTE5ZDU4ODIzMDQyMzk3NWRkN2M1ODE4MjU2NjlkNzUxZWMyZjBkYWIzZTdjMmM3ZTMxZTNjZTdmOGY4NzExOTQmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0JmFjdG9yX2lkPTAma2V5X2lkPTAmcmVwb19pZD0wIn0.Ho32ZZrSNpE4KVdpUr6EYB_DqMCnpyWxJYg-zaGfwxA "sample of dataset.")


* Our fusion technique results

* ![Fusion results.](https://private-user-images.githubusercontent.com/71946733/288082817-e5786f19-1450-4372-8238-4cfb5f7bb85d.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTEiLCJleHAiOjE3MDE3ODg5MTAsIm5iZiI6MTcwMTc4ODYxMCwicGF0aCI6Ii83MTk0NjczMy8yODgwODI4MTctZTU3ODZmMTktMTQ1MC00MzcyLTgyMzgtNGNmYjVmN2JiODVkLnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFJV05KWUFYNENTVkVINTNBJTJGMjAyMzEyMDUlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjMxMjA1VDE1MDMzMFomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTc4NWI5OTNhY2RmOWE3M2Q1MjgyZDZkNWYyNTU1MTZhMTQxOWQ4MGU0YTgxNzg1MWQwNGMzZGQ3YmZmZjAzMzEmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0JmFjdG9yX2lkPTAma2V5X2lkPTAmcmVwb19pZD0wIn0.6sJ1LBzE4WWPB2qk3AegAaSJiCyu9HcbvxkHDEHMYu0 "Fusion results.")


