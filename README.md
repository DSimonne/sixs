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

## Install PyNX
* Send a thank you email to Vincent Favre-Nicolin =D
* `cd /Packages`
* `mkdir PyNX_install`
* `cd PyNX_install/`
* `curl -O http://ftp.esrf.fr/pub/scisoft/PyNX/pynx-devel-nightly.tar.bz2`      # Installation details within install-pynx-venv.sh
* `source_p9`
* `pip install pynx-devel-nightly.tar.bz2[cuda,gui,mpi]`                        # Install avec les extras cuda, mpi, cdi
* cite `PyNX: high-performance computing toolkit for coherent X-ray imaging based on operators is out: J. Appl. Cryst. 53 (2020), 1404`, also available as `arXiv:2008.11511`

## Install bcdi
* Send a thank you email to Jerome Carnis =D
* `cd /Packages`
* `git clone https://github.com/carnisj/bcdi.git`
* `cd bcdi`
* `source_p9`
* `pip install .`
* cite `DOI: 10.5281/zenodo.3257616`

## Install facet-analyser (Debian 11 only)
* Send a thank you email to Fred Picca =D
* `cd /Packages`
* `git clone https://salsa.debian.org/science-team/facet-analyser.git`
* `cd facet-analyser`
* `sudo mk-build-deps -i`
* `debuild -b`
* `sudo debi`
* The package is now installed. You can check the locations of its files with the command `dpkg -L facet-analyser`
* You should see a file named `/usr/lib/x86_64-linux-gnu/paraview-5.9/plugins/FacetAnalyser/FacetAnalyser.so`
* Now launch `/usr/bin/paraview` (if not installed yet, good luck, refer to `https://www.paraview.org/Wiki/ParaView:Build_And_Install#Installing`)
* In paraview, go to Tools > Manage Plugins > Load New
* Here type the path to the plugin that was printed with the `dpkg -L facet-analyser` command.
* Feel free to add it to `/usr/bin/plugin` so that it is loaded automatically.
* cite `Grothausmann, R. (2015). Facet Analyser : ParaView plugin for automated facet detection and measurement of interplanar angles of tomographic objects. March.`

# Clusters at ESRF

## Firewall
cmd:
`ssh -X -p 5022 <login>@firewall.esrf.fr`

NoMachine:
`ssh -X -p 5622 <login>@firewall.esrf.fr`

## LID01
`ssh -X <login>@lid01gpu1`

`cd /data/id01/inhouse/david/`

 BCDI | PyNX
------------ | -------------
`source /data/id01/inhouse/richard/bcdiDevel.debian9/bin/activate` | `source /sware/exp/pynx/devel.debian9/bin/activate`


## rnice9
`ssh -X <login>@rnice9`

 BCDI | Paraview
------------ | -------------
`conda_bcdi (alias conda activate rnice.BCDI` | `source /sware/exp/paraview/envar.sh`

## slurm
`ssh -X <login>@slurm-nice-devel`

Demande GPU

`srun -N 1 --partition=p9gpu --gres=gpu:1 --time=01:00:00 --pty bash`

 BCDI | PyNX
| ------------ | ------------- |
| Frequently updated: `source /data/id01/inhouse/david/py38-env/bin/activate` | p9.devel (official branch) `source /sware/exp/pynx/devel.p9/bin/activate` |
| | p9, own environment `source /data/id01/inhouse/david/py38-env/bin/activate` |

### Kernels on slurm
* python3: all but PyNX
* py38-env : optimised for BCDI, phdutils and PyNX
* p9.pynx-devel : fonctionne pour pynx, frequently updated : `source /sware/exp/pynx/devel.p9/bin/activate`
* p9.pynx-gap : ?



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