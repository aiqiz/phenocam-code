# Phenocam Code

A collection of scripts and tools for processing PhenoCam images, analyzing vegetation indices, and understand seasonal index (start of season, end of season, and length of season). The processing workflow include both using auto-labeling and manual-labeling

---

## 📂 Project Structure
```bash
phenocam-code/
├── Auto_label/
│   ├── 1_auto_label_gcc_extraction.py          # auto labelling and extraction from time series of images
│   ├── Traffic Camera List - 4326.csv          # traffic camera location and index for city of toronto
│   └── 2_plot_raw_curve/
│       ├─ plot_one_month.py                    # plot raw gcc file for one month for one site
│       └──plot_multiple_month.py               # plot raw gcc file for one year for one site
│   └── 3_key_index_extract/
│       └──map_prepare.py                       # prepare each site's full year normalized gcc time series with key index
│   └── 4_plot_map/
│       ├─ metrics_map.py                       # plot map of sos/eos/los/peakdate with color bar
│       └──gif_continuous_field_map.py          # use interpolation to transfer discrete sites measurement to continous map of gcc flow
├── Manual_lable/
│   ├── manual_label.ipynb                      # jupyter notebook for explaining the manual labelling code in detail
│   └── full_code/
│      └── 2_plot_raw_curve/
│           ├─ plot_one_month.py                    # plot raw gcc file for one month for one site
│           └──plot_multiple_month.py               # plot raw gcc file for one year for one site
│   └── figures/
└── README.md           # Project documentation

---

## 🔧 Installation
Clone this repo:
```bash
git clone https://github.com/YOUR_USERNAME/phenocam-code.git
cd phenocam-code



