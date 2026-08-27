from setuptools import setup, find_packages

setup(
    name="prediccion_seleccion_pasantes",
    version="0.1.0",
    description="Modelo de clasificacion para predecir la seleccion de pasantes (TechNova Solutions)",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
)
