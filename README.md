# Welcome 

## This github is focused on the scripts that will be used during my thesis

Contact : david.simonne@universite-paris-saclay.fr

You can install phdutils via the setup.py script, so that you can use via a package after in python, see below

The bcdi subfolder gives some guidelines into how to process BCDI data

# Installing different packages

First, I advise you to create a Packages directory to keep these.
Secondly, I advise you to create a virtual environment to jelp with debogging, and so that once everything works, you don't update a package by mistake. To do so please follow the following steps:

## Create a virtual environment

* `mkdir py38-env`
* `cd py38-env/`
* `python3.8 -m venv .`
* `source bin/activate` # To activate the environment

Then you should create an alias such as: `alias source_p9="source /data/id01/inhouse/david/py38-env/bin/activate"`

## Install phdutils
* `cd /Packages`
* `git clone https://github.com/DSimonne/phdutils.git`
* `cd phdutils`
* `source_p9`
* `pip install .`

# Cluster at SOLEIL

## sixs3
`ssh -X sixs3`

`df -h` (pour voir les disques accessibles)

`cd /nfs/ruche-sixs/sixs-soleil/com-sixs/David`

 BCDI | PyNX
------------ | -------------
Installed on python3 | Installed on python3

### IPython sur ligne

`xcat`

### Macros

`/nfs/ruche-sixs/sixs-soleil/com-sixs/2021/Shutdown5/test/`

On peut créer des macros et les lancer avec `do.run("<filename>")`
Ils doivent respecter la syntaxe python et se lisent ligne par ligne
Possible d'éditer les scripts de la ligne dans `cd python/`


### RGA diagram
![mass_flow](https://user-images.githubusercontent.com/51970962/150782601-01500902-614c-4bd3-bfed-7ea41dfe1cc8.png)