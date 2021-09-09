# BCDI and ipywidgets

## To do
* use handlers to disable widgets that are for example useless with SIXS data
* automate the architecture specified in `https://github.com/DSimonne/phdutils/tree/master/phdutils/bcdi/`
* automate the rotation for SIXS data
* try to merge the notebooks of phase retrieval (`https://github.com/DSimonne/phdutils/tree/master/phdutils/bcdi/
PhasingNotebook.ipynb `) and facet analysis (`https://github.com/DSimonne/phdutils/tree/master/phdutils/bcdi/CompareFacetsEvolution.ipynb`) with the GUI
* merge the solutions for data visualisation in jupyter notebook(`https://github.com/DSimonne/phdutils/tree/master/phdutils/bcdi/plot.py`)
* Create a Dataset object that would used as iterable for the gui class, its attribute would then be all the parameters used for preprocessing, phase retrieval, facet retrieval, so that an external scientist could easily visualize the parameters used in the workflow

## How to use
* Connect to slurm, make sure that you do not need passwords when connecting for the batch scripts (see `http://www.linuxproblem.org/art_9.html`)
* Add this environemnet to your jupyter kernels: `source /data/id01/inhouse/david/py38-env/bin/activate`
* Open a jupyter notebook with this kernel and then run:

`from phdutils.bcdi.gui.gui import Interface`
`Test = Interface()`

## Guidelines for widgets
* Always press enter after editing a widget
* If somehow widgets are disabled when they should not be, please rerun the Interface cell and raise an issue with the details of each step that was undertaken, so that I can correct the bog
* You can also assign widgets values outside the interface, it will automaticcally update the widget, to do so, check the following code example:

```
specfile = "BCDI_2021_09_06_093352"
# specfile = "BCDI_2021_09_05_144113"
# specfile = "BCDI_2021_09_02_145714"
# specfile = "BCDI_2021_09_02_203654"
# 
Test._list_widgets_init.children[2].value = 242
Test._list_widgets_init.children[3].value = f"/data/visitor/hc4534/id01/B8_S1_P2/{specfile}/"
Test._list_widgets_init.children[4].value = "/data/id01/inhouse/david/ID01_September_2021/450/CO2/III_C17/"


Test._list_widgets_preprocessing.children[36].value = "Maxipix"
Test._list_widgets_preprocessing.children[42].value = ""
Test._list_widgets_preprocessing.children[44].value = f"/data/visitor/hc4534/id01/B8_S1_P2/{specfile}/mpx/data_mpx4_%05d.edf"

Test._list_widgets_preprocessing.children[1].value = "ID01"
Test._list_widgets_preprocessing.children[7].value = f"spec/{specfile}"
Test._list_widgets_preprocessing.children[8].value = "outofplane"

Test._list_widgets_preprocessing.children[52].value = "(-0.0011553664, 0, 0)"
Test._list_widgets_preprocessing.children[53].value = "0.60"
Test._list_widgets_preprocessing.children[54].value = "12994"
```

The widgets number might change with the evolution code, you can try to find the good one by playing with the different _list_widgets attributes, (should rename to list_widgets only)

### Dev
* disable all when preprocess is on otherwise it runs again, same for pynx, psf better handler
* images non sauvegardee pour preprocess, vient de bcdi.gu
* better interatcion between tabs if problem appears
* nest tab
* si long pour correct
* make sure that bcdi works also for script and then save commit
* select environment possibility
* netter separate all reconstructions
* catch all control c
* multiple plot comparison for rocking curves
* add inplane outofplane widget in correct and strain

Some screenshots of the work so far:
![Tab1](https://user-images.githubusercontent.com/51970962/130641516-ffe670b1-7b72-4b86-bef4-3b8bf4b7a797.png)
![Tab2](https://user-images.githubusercontent.com/51970962/130641522-9801d342-a1cc-4e87-8cb6-76cd78c909d3.png)
![Tab3](https://user-images.githubusercontent.com/51970962/130641578-f2515a53-09ba-47ac-a08e-cf093647d517.png)
![Tab4](https://user-images.githubusercontent.com/51970962/130641621-f6fafbaf-ac05-49e2-b9b5-e3ee2373b9e0.png)
![Tab5](https://user-images.githubusercontent.com/51970962/130641630-80fca919-ebb6-4ece-8638-95bbfd8a3dd3.png)
![Tab6](https://user-images.githubusercontent.com/51970962/130641638-9d59df04-2e60-495a-9de4-fcc0c3dfb9fe.png)
![Tab7](https://user-images.githubusercontent.com/51970962/130641648-48aaf34e-e70f-42f7-8a14-e283c519759e.png)
![Tab8](https://user-images.githubusercontent.com/51970962/130641650-62abc8d6-c45e-46ab-902e-d8a1211774ba.png)
![Output1](https://user-images.githubusercontent.com/51970962/130641658-20c82525-6a87-4414-baba-30defcba4328.png)
![Output2](https://user-images.githubusercontent.com/51970962/130641661-31ab2181-c1d4-4b24-89ed-8e4e2f15c5ca.png)