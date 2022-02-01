All tests were performed on Cloud7 @ 420 MHz

## Weak scaling

```python
ndpu = 2524
nfeatures = 16
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>DPU accuracy</th>
      <th>CPU accuracy</th>
      <th>Build time on DPU</th>
      <th>Build time on CPU</th>
      <th>Total time on DPU</th>
      <th>Total time on CPU</th>
    </tr>
    <tr>
      <th># of points per DPU</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>100</th>
      <td>0.791486</td>
      <td>0.796644</td>
      <td>1.484633</td>
      <td>0.315973</td>
      <td>3.817492</td>
      <td>0.341401</td>
    </tr>
    <tr>
      <th>300</th>
      <td>0.787198</td>
      <td>0.783469</td>
      <td>1.606975</td>
      <td>1.439782</td>
      <td>3.977112</td>
      <td>1.526295</td>
    </tr>
    <tr>
      <th>1000</th>
      <td>0.779131</td>
      <td>0.785513</td>
      <td>1.717301</td>
      <td>6.006987</td>
      <td>4.368524</td>
      <td>6.338849</td>
    </tr>
    <tr>
      <th>3000</th>
      <td>0.774060</td>
      <td>0.785403</td>
      <td>1.973278</td>
      <td>26.953712</td>
      <td>5.718544</td>
      <td>28.219126</td>
    </tr>
    <tr>
      <th>10000</th>
      <td>0.755640</td>
      <td>0.781600</td>
      <td>2.651749</td>
      <td>83.310026</td>
      <td>9.298141</td>
      <td>87.303477</td>
    </tr>
    <tr>
      <th>30000</th>
      <td>0.771507</td>
      <td>0.788651</td>
      <td>4.238167</td>
      <td>344.101506</td>
      <td>24.776381</td>
      <td>358.274337</td>
    </tr>
    <tr>
      <th>100000</th>
      <td>0.743415</td>
      <td>0.784216</td>
      <td>9.368356</td>
      <td>1388.148770</td>
      <td>83.270331</td>
      <td>1440.091028</td>
    </tr>
    <tr>
      <th>300000</th>
      <td>0.766377</td>
      <td>0.778323</td>
      <td>25.549841</td>
      <td>4113.248340</td>
      <td>237.159447</td>
      <td>4276.846957</td>
    </tr>
  </tbody>
</table>
</div>



## Strong scaling

```python
npoints_per_dpu = 100000
nfeatures = 16
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>DPU accuracy</th>
      <th>CPU accuracy</th>
      <th>Build time on DPU</th>
      <th>Build time on CPU</th>
      <th>Total time on DPU</th>
      <th>Total time on CPU</th>
    </tr>
    <tr>
      <th># of DPUs</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>10</th>
      <td>0.784685</td>
      <td>0.793027</td>
      <td>8.331406</td>
      <td>2.123142</td>
      <td>9.497109</td>
      <td>2.247365</td>
    </tr>
    <tr>
      <th>30</th>
      <td>0.770249</td>
      <td>0.781854</td>
      <td>8.370584</td>
      <td>8.529515</td>
      <td>11.258433</td>
      <td>8.934223</td>
    </tr>
    <tr>
      <th>100</th>
      <td>0.763350</td>
      <td>0.784064</td>
      <td>8.428060</td>
      <td>33.047634</td>
      <td>14.261116</td>
      <td>34.487501</td>
    </tr>
    <tr>
      <th>300</th>
      <td>0.769947</td>
      <td>0.783271</td>
      <td>8.428272</td>
      <td>113.963729</td>
      <td>18.101007</td>
      <td>118.712602</td>
    </tr>
    <tr>
      <th>1000</th>
      <td>0.767364</td>
      <td>0.789999</td>
      <td>8.822729</td>
      <td>437.248387</td>
      <td>35.240995</td>
      <td>453.881379</td>
    </tr>
    <tr>
      <th>2524</th>
      <td>0.755353</td>
      <td>0.784216</td>
      <td>9.591765</td>
      <td>1168.908268</td>
      <td>76.263814</td>
      <td>1212.420133</td>
    </tr>
  </tbody>
</table>
</div>



## Higgs Boson

11 M points (500 k set apart for test)  
27 features




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>DPU accuracy</th>
      <th>CPU accuracy</th>
      <th>Build time on DPU</th>
      <th>Build time on CPU</th>
      <th>Total time on DPU</th>
      <th>Total time on CPU</th>
    </tr>
    <tr>
      <th># of DPUs</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>100</th>
      <td>0.635726</td>
      <td>0.654442</td>
      <td>14.677905</td>
      <td>67.415419</td>
      <td>25.403798</td>
      <td>68.795795</td>
    </tr>
    <tr>
      <th>300</th>
      <td>0.653010</td>
      <td>0.654442</td>
      <td>5.892312</td>
      <td>67.415419</td>
      <td>9.739793</td>
      <td>68.795795</td>
    </tr>
    <tr>
      <th>1000</th>
      <td>0.653494</td>
      <td>0.654442</td>
      <td>3.403232</td>
      <td>67.415419</td>
      <td>6.337554</td>
      <td>68.795795</td>
    </tr>
    <tr>
      <th>2524</th>
      <td>0.668854</td>
      <td>0.654442</td>
      <td>3.230589</td>
      <td>67.415419</td>
      <td>7.133264</td>
      <td>68.795795</td>
    </tr>
  </tbody>
</table>
</div>


