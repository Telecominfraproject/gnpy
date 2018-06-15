
How to prepare the Excel input file
-----------------------------------

`examples/transmission_main_example.py <examples/transmission_main_example.py>`_ gives the possibility to use an excel input file instead of a json file. The program then will generate the corresponding json file for you.

The file named 'meshTopologyExampleV2.xls' is an example.

In order to work the excel file MUST contain at least 2 sheets:

- Nodes
- Links

(In progress) The File MAY contain an additional sheet:

- Eqt

Nodes sheet
-----------

Nodes sheet contains seven columns.
Each line represents a 'node' (ROADM site or an in line amplifier site ILA)::

  City (Mandatory) ; State ; Country ; Region ; Latitude ; Longitude ; Type

- **City** is used for the name of a node of the digraph. It accepts letters, numbers,commas,underscore,dash, blank... (not exhaustive).

   **City name MUST be unique** 

- **Type** is not mandatory. 

  - If not filled, it will be interpreted as an 'ILA' site if node degree is 2 and as a ROADM otherwise.
  - If filled, it can take "ROADM", "FUSED" or "ILA" values. If another string is used, it will be considered as not filled. FUSED means that spans ingress and egress spans will be fuesed together.  

- *State*, *Country*, *Region* are not mandatory.
  "Region" is a holdover from the CORONET topology reference file `CORONET_Global_Topology.xls <examples/CORONET_Global_Topology.xls>`_. CORONET separates its network into geographical regions (Europe, Asia, Continental US.) This information is not used by gnpy.

- *Longitude*, *Latitude* are not mandatory. If filled they should contain numbers.

**There MUST NOT be empty line(s) between two nodes lines**


Links sheet
-----------

Links sheet must contain sixteen columns::

                   <--           east cable from a to z                                   --> <--                  west from z to                                   -->
   NodeA ; NodeZ ; Distance km ; Fiber type ; Lineic att ; Con_in ; Con_out ; PMD ; Cable Id ; Distance km ; Fiber type ; Lineic att ; Con_in ; Con_out ; PMD ; Cable Id


Links sheets MUST contain all links between nodes defined in Nodes sheet.
Each line represents a 'bidir link' between two nodes. The two direction are represented on a single line with "east cable from a to z" fields and "west from z to a" fields. Values for 'a to z' may be different from values from 'z to a'. 
'z to a' direction MUST NOT be repeated in a different line.


Parameters for "east cable from a to z" and "west from z to a" are detailed in 2x7 columns. If not filled, "west from z to a" is copied from "east cable from a to z".

For example, a line filled with::

  node6 ; node3 ; 80 ; SSMF ; 0.2 ; 0.5 ; 0.5 ; 0.1 ; cableB ;  ;  ; 0.21 ; 0.2 ;  ;  ;  

will generate a unidir fiber span from node6 to node3 with::
 
  [node6 node3 80 SSMF 0.2 0.5 0.5 0.1 cableB] 

and a fiber span from node3 to node6::

 [node6 node3 80 SSMF 0.21 0.2 0.5 0.1 cableB] attributes. 

- **NodeA** and **NodeZ** are Mandatory. 
  They are the two endpoints of the link. They MUST contain a node name from the **City** names listed in Nodes sheet.

- **Distance km** is not mandatory. 
  It is the link length.

  - If filled it MUST contain numbers. If empty it is replaced by a default "80" km value. 
  - If value is below 150 km, it is considered as a single (bidirectional) fiber span.
  - If value is over 150 km the `transmission_main_example.py <examples/transmission_main_example.py>`_ program will automatically suppose that intermediate span description are required and will generate fiber spans elements with "_1","_2", ... trailing strings which are not visible in the json output.

- **Fiber type** is not mandatory. 

  If filled it must contain types listed in `eqpt_config.json <examples/eqpt_config.json>`_ in "Fiber" list "type_variety".
  If not filled it takes "SSMF" as default value.

- **Lineic att** is not mandatory. 

  It is the lineic attenuation expressed in dB/km.
  If filled it must contain positive numbers.
  If not filled it takes "0.2" dB/km value

- *Con_in*, *Con_out* are not mandatory and are not used yet. 

  They are the connector loss in dB at ingress and egress of the fiber spans.
  If filled they must contain positive numbers.
  If not filled they take "0.5" dB default value.

- *PMD* is not mandatory and and is not used yet. 

  It is the PMD value of the link in ps.
  If filled they must contain positive numbers.
  If not filled, it takes "0.1" ps value.

- *Cable Id* is not mandatory. 
  If filled they must contain strings with the same constraint as "City" names.




(in progress)

Eqpt sheet 
----------

Eqt sheet is optional. It lists the amplifiers types and characteristics on each degree of the *Node A* line.
Links sheet must contain sixteen columns::

                   <--           east cable from a to z        --> <--        west from z to a                 -->
  Node A ; Node Z ; amp type ; att_in ; amp gain ; tilt ; att_out ; amp type ; att_in ; amp gain ; tilt ; att_out

If the sheet is present, it MUST have as many lines as egress directions of ROADMs defined in Links Sheet. 

For example, consider the following list of links (A,B and C being a ROADM and amp# ILAs)

::

  A    - amp1
  amp1 - amp2
  Amp2 - B
  A    - amp3
  amp3 - C

then Eqpt sheet should contain:

::

  A    - amp1
  amp1 - amp2
  Amp2 - B
  A    - amp3
  amp3 - C
  B    - amp2
  C    - amp3


`create_eqpt_sheet.py <examples/create_eqpt_sheet.py>`_ helps you to create a template for the mandatory entries of the list.

.. code-block:: shell

    $ cd examples
    $ python create_eqpt_sheet.py meshTopologyExampleV2.xls


- **Node A** is mandatory. It is the name of the node (as listed in Nodes sheet).
  If Node A is a 'ROADM' (Type attribute in sheet Node), its number of occurence must be equal to its degree.
  If Node A is an 'ILA' it should appear only once.

- **Node Z** is mandatory. It is the egress direction from the *Node A* site. Multiple Links between the same Node A and NodeZ is not supported.

- **amp type** is not mandatory. 
  If filled it must contain types listed in `eqpt_config.json <examples/eqpt_config.json>`_ in "Edfa" list "type_variety".
  If not filled it takes "std_medium_gain" as default value.

- **amp_gain** is not mandatory. It is the value to be set on the amplifier (in dB).
  If not filled, it will be determined with design rules in the convert.py file.
  If filled, it must contain positive numbers.

- *att_in* and *att_out* are not mandatory and are not used yet. They are the value of the attenautor at input and output of amplifier (in dB).
  If filled they must contain positive numbers.

- *tilt* --TODO--

# to be completed #

(in progress)

Service sheet 
-------------

Service sheet is optional. It lists the services for which path and feasibility must be computed with path_requests_run.py.

Service sheet must contain 11 columns::  

   route id ; Source ; Destination ; TRX type ; Mode ; System: spacing ; System: input power (dBm) ; System: nb of channels ;  routing: disjoint from ; routing: path ; routing: is loose?

- **route id** is mandatory. It must be unique. It is the identifier of the request. It must be an integer value (TODO: relax this format in future development to accept any strings)

- **Source** is mandatory. It is the name of the source node (as listed in Nodes sheet). Source MUST be a ROADM node. (TODO: relax this and accept trx entries)

- **Destination** is mandatory. It is the name of the destination node (as listed in Nodes sheet). Source MUST be a ROADM node. (TODO: relax this and accept trx entries)

- **TRX type and mode** are mandatory. They are the variety type and selected mode of the transceiver to be used for the propagation simulation. These modes MUST be defined in the equipment library. The format of the mode is used as the name of the mode. (TODO: maybe add another  mode id on Transceiver library ?). In particular the mode selection defines the channel baudrate to be used for the propagation simulation. 

- **System: spacing ; System: input power (dBm) ; System: nb of channels** are mandatory input defining the system parameters for the propagation simulation.

  - spacing is the channel spacing defined in GHz
  - input power is the channel optical input power in dBm
  - nb of channels is the number of channels to be used for the simulation.

- **routing: disjoint from ; routing: path ; routing: is loose?** are optional.

  - disjoint from: (work not started) identifies the requests from which this request must be disjoint. It is not used yet 
  - path: is the set of ROADM nodes that must be used by the path. It must contain the list of ROADM names that the path must cross. TODO : only ROADM nodes are accepted in this release. Relax this with any type of nodes.
  - is loose?  (in progress) 'no' value means that the list of nodes should be strictly followed, while any other value means that the constraint may be relaxed if the node is not reachable. 

# to be completed #

convert_service_sheet.py
------------------------


`convert_service_sheet.py <examples/convert_service_sheet.py>`_ converts the service sheet to a json file following the Yang model for requesting Path Computation defined in `draft-ietf-teas-yang-path-computation-01.txt <https://www.ietf.org/id/draft-ietf-teas-yang-path-computation-01.pdf>`_. TODO: verify that this implementation is correct + give feedback to ietf on what is missing for our specific application.
For PSE use, additional fields with trx type and mode have been added to the te-bandwidth field.

**Usage**: convert_service_sheet.py [-h] [-v] [-o OUTPUT] [workbook_name.xls]

.. code-block:: shell

    $ cd examples
    $ python convert_service_sheet.py meshTopologyExampleV2.xls -o service_file.json

-o output_file.json is an optional parameter: 

  - if not used, the program output the json data on  standard output and on a json file with name 'workbook_name_services.json'.

A template for the json file can be found here: `service_template.json <service_template.json>`_

path_requests_run.py
------------------------

**Usage**: path_requests_run.py [-h] [-v] [-o OUTPUT]
                            [network_filename xls or json] [service_filename xls or json] [eqpt_filename json]

.. code-block:: shell

    $ cd examples
    $ python path_requests_run.py meshTopologyExampleV2.xls service_file.json eqpt_file -o output_file.json

A function that computes performances for a list of services provided in the service file (accepts json or excel format.

If no output file is given, the computation is shown on standard output for demo.
If a file is specified with the optional -o argument, the result of the computation is converted into a json format following  the Yang model for requesting Path Computation defined in `draft-ietf-teas-yang-path-computation-01.txt <https://www.ietf.org/id/draft-ietf-teas-yang-path-computation-01.pdf>`_. TODO: verify that this implementation is correct + give feedback to ietf on what is missing for our specific application.

A template for the result of computation json file can be found here: `path_result_template.json <path_result_template.json>`_

Important note: path_requests_run.py is not a dimensionning tool, it only computes path feasibility assuming full load with system parameters input in the service file. The network topology is created only once for the set of requests. As a result the transceiver element acts as a "logical starting/stopping point" for the spectral information propagation. At that point it is not meant to represent the capacity of add drop ports
As a result transponder type is not part of the network info. it is related to the list of services requests.

(in progress)


