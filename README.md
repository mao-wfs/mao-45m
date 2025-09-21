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
