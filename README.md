# mao-45m
MAO controller for the Nobeyama 45m telescope

## Command line interface

### Send existing VDIF file over UDP multicast

```shell
mao-45m vdif send /path/to/vdif <options>
```

See `mao-45m vdif send --help` for more information.

### Receive VDIF file over UDP multicast

```shell
mao-45m vdif receive /path/to/vdif <options>
```

See `mao-45m vdif receive --help` for more information.

### Send subreflector parameters to COSMOS over TCP

```shell
mao-45m cosmos send --dX 1.0 --dZ 2.0 <options>
```

See `mao-45m cosmos send --help` for more information.

### Receive current state from COSMOS over TCP

```shell
mao-45m cosmos receive <options>
```

See `mao-45m cosmos receive --help` for more information.


## Feed models

### 2024

```csv
feed, position_radius, position_angle, homologous_EPL_A, homologous_EPL_B, homologous_EPL_C, EPL_over_dZ, EPL_over_dX
# str, float[m], float[deg], float[m], float[deg], float[m], float[1], float[1]
c, 5.00, 92.00, 0.052957, +37.508, -0.011955, 1.800, -0.2690
t, 16.2, 72.00, 0.095264, +67.690, +0.028751, 1.579, -0.6940
r, 16.2, 0.000, 0.039385, +4.5560, -0.028109, 1.583, +0.0060
b, 16.2, 268.0, 0.098803, -63.839, -0.090363, 1.587, +0.7360
l, 16.2, 180.0, 0.039617, +1.4710, -0.029742, 1.598, -0.0017
```
