# Tutorial on mass flow controller (XCAT) and mass flow spectrometer (RGA)
Call the floor coordinator (9797) and tell him that we open the toxic gaz bottles to start the red experiment
SAME WHEN WE CLOSE THEM

# Beginning
* Create a folder in reactor_data/ for each experiment (e.g first day : reactor_data/heating_no_gaz)
* One RGA data file for each experiment, the XCAT data file can last for longer, the two files time stamps issues, so to analyze the data, I advise to use the scripts in this repository, to be sure that they are linked.

# The Residual Gas Analyser (RGA)
* Acts as a probe for species present in sample atmospheres
* One file per experiment (e.g. heating and cooling of sample under same variating gaseous environment)
* Generates data like this :

```
Jan 20, 2021  10:49:46 AM
Residual Gas Analyzer Software 
RGA Software Version, 3.213.006

Pressure vs Time Scan Setup:

Active channels in Scan, 6 
Units, Torr
Sample Period, 1.00, sec
Focus Voltage, 90, Volts 
Ion Energy, HIGH 
Electron Energy, 70, eV 
CEM Voltage, 1060, Volts
CEM Gain, 1.44E+003 
Sensitivity Factor, 1.35E-004 
Filament Current, 1.00, mAmps
Start time, Jan 20, 2021  10:49:46 AM 
Channel,  Mass(amu),     Name,                 Cal Factor,  Noise Floor, CEM Status

1         28.00          CO                    1.00         3            OFF
2         44.00          CO2                   1.00         3            OFF
3         32.00          O2                    1.00         3            OFF
4         40.00          Argon                 1.00         3            OFF
5         2.00           H2                    1.00         3            OFF
6         18.00          Water                 1.00         3            OFF


Time(s)      Channel#1   Channel#2   Channel#3   Channel#4   Channel#5   Channel#6   

0.000,   7.0732E-008,   9.4566E-008,   4.5203E-009,   7.1560E-009,   1.6492E-007,   1.1405E-006,  
1.934,   6.9779E-008,   9.4984E-008,   4.7975E-009,   7.0488E-009,   1.6416E-007,   1.1334E-006,  
3.946,   6.8226E-008,   9.2667E-008,   4.7768E-009,   7.0880E-009,   1.6242E-007,   1.1277E-006,  
6.021,   6.8489E-008,   9.2420E-008,   4.5336E-009,   7.3688E-009,   1.6136E-007,   1.1256E-006,  
8.065,   6.8092E-008,   9.2156E-008,   4.8411E-009,   7.3067E-009,   1.6051E-007,   1.1131E-006,  
```

* As many columns (channels) as gases we gave in input, be careful to include all the gases before launching an experiment since they are not automatically detected
* Click on the green button to initiate data acquisition, and on the red button to stop
* **IMPORTANT** Save data as `.rga` and then **export as ascii to save as `.txt`**
* Save in data_folder/xcat_data

# XCAT
* Stores the position of the valves as well as the setpoints for each gas, allows us to better understand the dynamics
* Generates data like this :

```
Time	NO flow	NO setpoint	NO valve	H2 flow	H2 setpoint	H2 valve	O2 flow	O2 setpoint	O2 valve	CO flow	CO setpoint	CO valve	Ar flow	Ar setpoint	Ar valve	Shunt flow	Shunt setpoint	Shunt valve	Reactor pressure	Reactor setpoint	Reactor valve	Shunt pressure	Shunt setpoint	Shunt valve	MIX valve	MRS valve	INJ valve	OUT valve
3693980302.604936	0.000000	0.000000	0.000000	3693980302.804947	0.000000	0.000000	0.000000	3693980303.004959	0.000000	0.000000	0.000000	3693980303.205970	0.000000	0.000000	0.000000	3693980303.406982	0.000000	0.000000	0.000000	3693980303.607994	2.756250	0.000000	0.000000	3693980303.811005	0.000844	0.000000	8.818333	3693980304.012016	0.000000	0.000000	8.818333	3693980302.404924	3.000000	6.000000
3693980304.114022	0.000000	0.000000	0.000000	3693980304.216028	0.000000	0.000000	0.000000	3693980304.316034	0.000000	0.000000	0.000000	3693980304.417039	0.000000	0.000000	0.000000	3693980304.517045	0.000000	0.000000	0.000000	3693980304.617051	1.104688	0.000000	0.000000	3693980304.719057	0.000844	0.000000	8.818333	3693980304.820063	0.000000	0.000000	8.818333	3693980304.014017	3.000000	6.000000
```
* Saved only locally on the computer inside the hutch, you need to retrieve them.
* If it crashes: datasocket server needs 2 connections, si pas de clavier, souris vers le haut

# Launching an experiment

## Macro

```
import GasTool as gt
import time
mv(heater3, 0)
gt.condE()
# 50ml/min 0.3Bar
# flow(Arg=32, COg=0, O2g=8,NOg=0, H2g=7.81, mixg=6, press=0.3 )
time.sleep(300)

gt.tloopFastOff(0,1.8,100)

time.sleep(180)
gt.stflow()
```

* Type `do.run("macro.txt")` in the MED env
* Make sure the rga is recording (green button, you should see the Argon)
* Save your macro in the directory where the scans are saved
* It will generate a log file (log_nameofmacro_date_time.txt) in folder, interesting to be sure at what time we changed conditions on the particle (gas or temperature), put them in data_folder/xcat_data/date/log

# Visualize data
* Copy the example notebook and use the fonctions present in xcat_scripts to visualize the data
* Respect the order
* Make sure to have copied the xcat_scripts folder too


# GasTool commands

`import GasTools as gt`

ALL COMMANDS ARE /home/python/GasTool.py
Once imported, you can have detail about the commands with `?`, e.g. `gt.closeall?`

## Start the flow at 0.5 bar in the reactor, 50 ml/min and only Argon
* `gt.stflow()`

## How to open the dome
* put Ar to 1bar to be able to open the dome (kill vacuum)
* `mv heater4, 0`
* `mv rea, 1`
* `gt.closeall()`

### Going back to AR
`gt.closeall()`
`gt.stflow()`

### Launching any command written in GasTool, e.G. condA
`gt.condA()`

### if toxic gases present, purge, also puts 1 bar of Ar
`mv heater4, 0`
`gt.wash()`
`gt.closeall()`


### LEAVE BEAMILNE ? SAFETY FIRST
CLOSE BOTTLES IN BLUE CABINET (first the valve in the bottle, then the lower pressure valve)
gt.pall() !! close bottles before!!! and dome closed
close the manivelle after everything is pumped
gt.whg()

## Windows crash
* `vncviewer pcxcat`

# Macros

We chose to do 3 SBS scan per temperature for each conditions, to make sure that we have consistent data
For each heater position, launch each conditions, wait first for 10 minuts, then launch three SBS scans, make sure to realign in between

## cond A : Ar : 49, O2 : 0, NH3 : 1

## cond B : Ar : 48.5, O2 : 0.5, NH3 : 1

## cond C : Ar : 48, O2 : 1, NH3 : 1

## cond D : Ar : 47, O2 : 2, NH3 : 1

7.81 NH3 -> 9 Ar et 1 NH3

# Ammonia oxidation reactions
4NH3 + 3O2 -> 6H2O + 2N2
4NH3 + 4O2 -> 6H2O + 2N2O
4NH3 + 5O2 -> 6H2O + 4NO