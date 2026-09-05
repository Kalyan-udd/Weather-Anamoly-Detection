Folder Structure: 

Weather_anamoly/
        ---.venv/
        ---artifacts/
            ---model.joblib
        ---logs/
            ---app.log
        ---model_training/
            ---data/
                ---weather_cache.sqlite
                ---__init__.py
                ---coordinates/
                    ---coordinates.sqlite
            ---__init__.py
            ---data_injestion.ipynb
            ---dataset_formation_trail1.ipynb
        ---src/
            ---weather_anomaly/
                ---__init__.py
                ---logger.py
                ---utils.py
            ---weather_anomaly_detection.egg-info
            ---__init__.py
        ---templates/
            ---*.html
            ---*.css
            ---*.js
        ---tests/
        ---Dockerfile
        ---.dockerignore
        ---.python-version
        ---.gitignore
        ---main.py
        ---pyproject.toml
        ---requirements.txt
        ---uv.lock




