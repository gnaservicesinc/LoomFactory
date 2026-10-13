# Categorizer asset

`DTTC3.mlmodel` is the user's trained diffuse-category classifier. Loom Factory compiles it as the initial categorizer for reviewing source images. It is a model asset, not reused application or trainer code.

Factory exports category folders for image-classifier training in Create ML. The imported classifier and its validated label catalog become part of the complete Texture Loom deployment package. Categorization can use that project classifier after import.
