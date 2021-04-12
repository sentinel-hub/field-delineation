//VERSION=3

//
// Copyright (c) Sinergise, 2019 -- 2021.
//
// This file belongs to subproject "field-delineation" of project NIVA (www.niva4cap.eu).
// All rights reserved.
//
// This source code is licensed under the MIT license found in the LICENSE
// file in the root directory of this source tree.
//

function setup() {
  return {
    input: [{
      bands: ["B02", "B03", "B04", "B08", "dataMask", "CLP"],
      units: "DN"
    }],
    output: [
      {id: "B02", bands: 1, sampleType: SampleType.UINT16},
      {id: "B03", bands: 1, sampleType: SampleType.UINT16},
      {id: "B04", bands: 1, sampleType: SampleType.UINT16},
      {id: "B08", bands: 1, sampleType: SampleType.UINT16},
      {id: "dataMask", bands: 1, sampleType: SampleType.UINT8},
      {id: "CLP", bands: 1, sampleType: SampleType.UINT8}
    ],
    mosaicking: Mosaicking.ORBIT
  }
}

function updateOutput(outputs, collection) {
    Object.values(outputs).forEach((output) => {
        output.bands = collection.scenes.length;
    });
}

function updateOutputMetadata(scenes, inputMetadata, outputMetadata) {
  let dates = []
  scenes.forEach(scene => {
    dates.push(scene.date);
  })
  outputMetadata.userData = {
    dates:  JSON.stringify(dates)
  };
}

function evaluatePixel(samples) {
  var n_observations = samples.length;
  let band_b02 = new Array(n_observations).fill(0);
  let band_b03 = new Array(n_observations).fill(0);
  let band_b04 = new Array(n_observations).fill(0);
  let band_b08 = new Array(n_observations).fill(0);
  let mask = new Array(n_observations).fill(0);
  let clp = new Array(n_observations).fill(0);

  samples.forEach((sample, index) => {
    band_b02[index] = sample.B02;
    band_b03[index] = sample.B03;
    band_b04[index] = sample.B04;
    band_b08[index] = sample.B08;
    mask[index] = sample.dataMask;
    clp[index] = sample.CLP;
  });

  return {
    B02: band_b02,
    B03: band_b03,
    B04: band_b04,
    B08: band_b08,
    dataMask: mask,
    CLP: clp
  };
}
