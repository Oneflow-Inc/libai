
## Steps for Build Docs

Before generating the doc, you need to install libai by:

```bash
cd ${libai-path}
pip install -e .
```

Then build docs by:

```bash
cd ${libai-path}/docs
pip install -r requirements.txt --user
make html
```


