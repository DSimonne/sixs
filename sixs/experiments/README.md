# Tutorial on mass flow controller (XCAT) and mass flow spectrometer (RGA)

* First, to begin the experiment, call the floor coordinator (9797) and tell him that you open the toxic gaz bottles to start the red experiment.
* You should **absolutely** tell them when you leave or when you close them because the experiment is finished.

# Beginning
* You should create a new RGA (mass spectrometer) data file for each experiment, the XCAT (mass flow controller) data file usually lasts for the whole experiment,
* The two files have time stamps issues used to analyze the data,
* I advise you to use the scripts in this repository, to be sure that they are correctly linked.

# The Residual Gas Analyser (RGA)

## Operating
* Acts as a probe for species present in the sample atmospheres.
* One file per experiment (e.g. heating and cooling of sample under same variating gaseous environment)
* Time shift between computers: (502 secondes in Jan 2022, please calibrate it for time-resolved exp.).
* There is a leak in the reactor cell that is directly connected to a ultra high vacuum chamber, in which the product detection happens.
* We are **not** measuring the pressure in the reactor cell but after the leak ! There must be a normalisation step to correctly find the pressure in the reactor chamber.
* The pressure in that chamber, ~ 3e-6, should be constant throughout the experiment, you can control it with a valve.
* The leak depends on the temperature (dilatation of the valve)
* No dependance on particle size (Avogadro nb)
* However the flux depends on the speed, thus on the mass.
* We can compute the dependance of the detector on the temperature by following its signal during a temperature cycle while keeping the reagent product constant. However, we are only sensible to the reagents.
* Pressure control is after the reactor chamber exit.
* For the small reactor, the leak is connected a meter after the chamber exit.

## Background signals
* There is always N2 in ultra-high vacuum chambers, as well as H2O.
* Possible to normalize by using the total pressure in the chamber.
* Possible also to divide the partial pressures collected by that of the carrier gas, that should be known  and stable at constant reaction.
* Then, keeping in mind that we have, e.g. 82% of Ar and that the total pressure is of 0.3 bar, we can fix its pressure to 0.82\*0.3 bar.
* Total pressure in the reactor is equal to the sum of the pressure of the reagents + carrier gas outside the reaction. Add the products when the reaction happens.
* For Ammonia, we must use either NH or NH3 when normalizing, we showed that NH3 is usually better.

## Data file example

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

* As many columns (channels) as gases we gave in input, be careful to include all the gases before launching an experiment since they are not automatically detected.
* Click on the green button to initiate data acquisition, and on the red button to stop.
* **IMPORTANT** Save data as `.rga` and then **save as ascii to save as `.txt`**, which is the readable file.

# XCAT

![image](https://user-images.githubusercontent.com/51970962/152839468-73ec67e8-9cdd-4aba-bf26-83f635241d57.png)

* Stores the position of the valves that control the gas flow, as well as the setpoints for each gas, allows us to better understand the dynamics.
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
* Make sure the RGA is recording (green button, you should see the Argon)
* Save your macro in the directory where the scans are saved
* It will generate a log file (log_nameofmacro_date_time.txt) in folder, interesting to be sure at what time we changed conditions on the particle (gas or temperature), put them in data_folder/xcat_data/date/log

# GasTool commands

`import GasTools as gt`

The module is in `/home/python/GasTool.py`. Once imported, you can have detail about each command with `?`, e.g. `gt.closeall?`

## Start the flow at 0.5 bar in the reactor, 50 ml/min and only Argon
* `gt.stflow()`

## Open the dome
* `gt.stflow(press=1)`
* `mv heater4, 0`
* `gt.closeall()`

### if toxic gases present, purge, also puts 1 bar of Ar
`mv heater4, 0`
`gt.wash()`
`gt.closeall()`


### LEAVE BEAMLINE ? SAFETY FIRST
CLOSE BOTTLES IN BLUE CABINET (first the valve in the bottle, then the lower pressure valve)
`gt.pall()` !! close bottles before!!! and dome closed
close the manivelle after everything is pumped
gt.whg()

## Windows crash
* `vncviewer pcxcat`

# Sputtering

* Make sure that the sample has the thermocouple connected to it !!
* Remove the detector nose
* Go to RT
* Turn off the RGA
* Close the bottle valves
* Empty the reactor gt.prea()
* Close the reactor valves
* Connect the multimeter to the sputtering current
* Open the reactor with the visseuse
* Wait for the pressure to go down
* Put Argon for sputtering (open chamber valve, already Argon inside)
* leak to ~ 2e-5 mBar
* Branch ion gun cable
* Screw three screws
* Turn on ion cannon
* Switch operate mode on
* By moving the reactor chamber up and down you must find the highest sputtering current

Settings are: 1.5 keV, 9µA current detected, for 30 min