# Welcome 

## This github is focused on the scripts that will be used during my thesis

Contact : david.simonne@synchrotron-soleil.fr

You can install phdutils via the setup.py script, so that you can use via a package after in python

The bcdi subfolder gives some guidelines into how to process BCDI data

# Clusters at ESRF

## Firewall
cmd:
`ssh -X -p 5022 <login>@firewall.esrf.fr`

NoMachine:
`ssh -X -p 5622 <login>@firewall.esrf.fr`

## ID01
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
`ssh -X <login>@slurm-access`

Demande GPU

`srun -N 1 --partition=p9gpu --gres=gpu:1 --time=01:00:00 --pty bash`

 BCDI | PyNX
------------ | -------------
No bcdi | `source /sware/exp/pynx/devel.p9/bin/activate`

### Kernels on slurm
* p9.widgets : optimisé pour utiliser les widgets et thorondor
* p9.bcdi : not existing yet, pb is the use of qt in bcdi, will correct soon
* p9.pynx-devel : fonctionne pour pynx : `source /sware/exp/pynx/devel.p9/bin/activate`
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