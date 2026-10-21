# Interactive voltage-cube viewer

The viewer displays the measured and fitted arrays used for Fig. 4a--d.
The channel-to-port/color convention is CH2/P1 = red, CH3/P2 = green, and CH4/P3 = blue.
`data/cube_data.json.gz` stores the full float64 data. Browser subsampling
changes display density only.

Serve this directory over HTTP and open `index.html` through the server. First generate the ignored local viewer dataset with
`python scripts/export_cube_viewer_data.py`. Then, from the repository root:

```bash
python3 -m http.server 8000 --bind 127.0.0.1
```

Then open `http://localhost:8000/viewer/`.
