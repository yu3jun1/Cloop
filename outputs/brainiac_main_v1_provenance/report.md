# CLARITY Loop report: brainiac_main_v1_provenance

This report separates factual forecasting, historical observed replay, and independent synthetic closed-loop evidence.

## Run and data audit

- Protocol: `main_v1`
- Data signature: `d9992774629eb993ca017f51cb140032a15b0dabd63e4aa2da937d4814af20b0`
- Project revision: `74c6e4b02dee6cf294cf3799dc09d5939af807bd`
- Test revealed: `True`
- Protocol frozen: `True`
- Cohort: `{"patients": 155, "split_counts": {"test": 24, "train": 108, "validation": 23}, "windows": {"test": {"H1": 63, "H2": 41, "H3": 25}, "train": {"H1": 251, "H2": 147, "H3": 75}, "validation": {"H1": 62, "H2": 39, "H3": 22}}}`
- Audit: `{"excluded_counts": {"duplicate_mri_day": 1, "excluded_unknown_interval": 16, "fewer_than_two_valid_times": 48, "invalid_time_interval": 1, "known_interval": 376, "survival_death_shifted_to_L_plus_1": 59, "survival_invalid_survival_time": 78, "uncertain_destination_event": 20}, "time_quality": {"timepoint_sources": {"imputed_interior": 1, "imputed_leading": 0, "imputed_trailing": 1, "observed": 546, "unknown": 0}, "total_timepoints": 548, "total_transitions": 393, "transitions_involving_imputation": 3, "transitions_involving_unknown_time": 0, "windows": {"H2": {"eligible": 227, "involving_imputation": 2, "involving_unknown_time": 0}, "H3": {"eligible": 122, "involving_imputation": 2, "involving_unknown_time": 0}}}, "warnings": []}`
- Data assumptions: `{"genomics_visibility": "structured genomics are treated as baseline-known because per-test dates are not available; this assumption must be disclosed", "progression_input": "disabled unless a reliable occurrence date is audited", "time_quality": "per-timepoint observed/imputed/unknown provenance is preserved and audited; TP numbering is never used as time"}`

## Completed experiment groups

- dynamics_done: `True`
- evaluated: `True`
- interrupted: `False`
- outcome_done: `True`
- prepared: `True`
- protocol_frozen: `True`

## Dynamics — factual actions and times

```json
{
  "test": {
    "baseline/17": {
      "CosSim@1": 0.5986625614856916,
      "CosSim@2": 0.5069076565162438,
      "CosSim@3": 0.4331202581524849,
      "long_mse": 1.03732515201336,
      "mse@1": 0.7511243564741952,
      "mse@2": 0.9202142772151203,
      "mse@3": 1.1544360268115996,
      "n@1": 63,
      "n@2": 41,
      "n@3": 25,
      "patient_macro_mse@1": 0.8456192995788474,
      "patient_macro_mse@2": 0.9656690896178286,
      "patient_macro_mse@3": 1.1908381987701764,
      "patients@1": 22,
      "patients@2": 16,
      "patients@3": 11
    },
    "baseline/29": {
      "CosSim@1": 0.6330614663246605,
      "CosSim@2": 0.5646538027539486,
      "CosSim@3": 0.5071353462338447,
      "long_mse": 1.1045050728902583,
      "mse@1": 0.7903217181326851,
      "mse@2": 0.9600115789145958,
      "mse@3": 1.248998566865921,
      "n@1": 63,
      "n@2": 41,
      "n@3": 25,
      "patient_macro_mse@1": 0.8898243533955378,
      "patient_macro_mse@2": 1.0022742810348668,
      "patient_macro_mse@3": 1.2497721621484466,
      "patients@1": 22,
      "patients@2": 16,
      "patients@3": 11
    },
    "baseline/7": {
      "CosSim@1": 0.6164656937831924,
      "CosSim@2": 0.5233673633235257,
      "CosSim@3": 0.4455375583469868,
      "long_mse": 1.0878216091307198,
      "mse@1": 0.7666567854938053,
      "mse@2": 0.9507734019581865,
      "mse@3": 1.2248698163032532,
      "n@1": 63,
      "n@2": 41,
      "n@3": 25,
      "patient_macro_mse@1": 0.8516788985015769,
      "patient_macro_mse@2": 0.9942128146067262,
      "patient_macro_mse@3": 1.2603233194712435,
      "patients@1": 22,
      "patients@2": 16,
      "patients@3": 11
    },
    "ensemble/17": {
      "CosSim@1": 0.6108488853252123,
      "CosSim@2": 0.5328774576688685,
      "CosSim@3": 0.49059246003627777,
      "long_mse": 0.8821550873430763,
      "mse@1": 0.7159880610212447,
      "mse@2": 0.8237566788022112,
      "mse@3": 0.9405534958839417,
      "n@1": 63,
      "n@2": 41,
      "n@3": 25,
      "patient_macro_mse@1": 0.792592005296187,
      "patient_macro_mse@2": 0.8655352105076115,
      "patient_macro_mse@3": 0.9322485517371785,
      "patients@1": 22,
      "patients@2": 16,
      "patients@3": 11
    },
    "ensemble/29": {
      "CosSim@1": 0.5963542110153607,
      "CosSim@2": 0.5202972087554816,
      "CosSim@3": 0.4763731718063354,
      "long_mse": 0.8820682597305717,
      "mse@1": 0.7260130830700435,
      "mse@2": 0.8253457415394667,
      "mse@3": 0.9387907779216766,
      "n@1": 63,
      "n@2": 41,
      "n@3": 25,
      "patient_macro_mse@1": 0.7969256328813956,
      "patient_macro_mse@2": 0.8610191135667264,
      "patient_macro_mse@3": 0.917694648106893,
      "patients@1": 22,
      "patients@2": 16,
      "patients@3": 11
    },
    "ensemble/7": {
      "CosSim@1": 0.6150840857908839,
      "CosSim@2": 0.5495683025659585,
      "CosSim@3": 0.5133410984277725,
      "long_mse": 0.8648796773029537,
      "mse@1": 0.7095490242280658,
      "mse@2": 0.8065680945064964,
      "mse@3": 0.923191260099411,
      "n@1": 63,
      "n@2": 41,
      "n@3": 25,
      "patient_macro_mse@1": 0.7924173500614635,
      "patient_macro_mse@2": 0.8446085260560114,
      "patient_macro_mse@3": 0.8959256162246068,
      "patients@1": 22,
      "patients@2": 16,
      "patients@3": 11
    },
    "paired_relative_improvement": {
      "baseline_vs_rrt": {
        "mean": 16.854213472674783,
        "n": 3,
        "per_seed": {
          "17": 15.533832264272641,
          "29": 20.33027582057949,
          "7": 14.69853233317221
        },
        "std": 3.0391920930758394
      },
      "baseline_vs_rrt_ensemble": {
        "mean": 26.803958215802663,
        "n": 3,
        "per_seed": {
          "17": 22.865901894809078,
          "29": 28.668057510295622,
          "7": 28.877915242303292
        },
        "std": 3.412070596779763
      },
      "ensemble_vs_rrt_ensemble": {
        "mean": 10.174182024840926,
        "n": 3,
        "per_seed": {
          "17": 9.29810280483831,
          "29": 10.679823845991196,
          "7": 10.54461942369327
        },
        "std": 0.7617126484065512
      }
    },
    "persistence": {
      "CosSim@1": 0.6474110973732812,
      "CosSim@2": 0.6171188394470912,
      "CosSim@3": 0.6034678369760513,
      "long_mse": 1.0402254805077868,
      "mse@1": 0.8202946836513186,
      "mse@2": 0.9391296929339084,
      "mse@3": 1.1413212680816651,
      "n@1": 63,
      "n@2": 41,
      "n@3": 25,
      "patient_macro_mse@1": 0.9219971892508593,
      "patient_macro_mse@2": 0.9779536625137553,
      "patient_macro_mse@3": 1.0982724392052852,
      "patients@1": 22,
      "patients@2": 16,
      "patients@3": 11
    },
    "rrt/17": {
      "CosSim@1": 0.630512446283348,
      "CosSim@2": 0.5736551190294871,
      "CosSim@3": 0.520405453145504,
      "long_mse": 0.8761888028644934,
      "mse@1": 0.6987236966452901,
      "mse@2": 0.7950493279026776,
      "mse@3": 0.9573282778263092,
      "n@1": 63,
      "n@2": 41,
      "n@3": 25,
      "patient_macro_mse@1": 0.8152764346224792,
      "patient_macro_mse@2": 0.8610242563299835,
      "patient_macro_mse@3": 0.97832911168084,
      "patients@1": 22,
      "patients@2": 16,
      "patients@3": 11
    },
    "rrt/29": {
      "CosSim@1": 0.6321455320668599,
      "CosSim@2": 0.5658715850696331,
      "CosSim@3": 0.5114434438943863,
      "long_mse": 0.8799561451193763,
      "mse@1": 0.6905967349570895,
      "mse@2": 0.796896772050276,
      "mse@3": 0.9630155181884765,
      "n@1": 63,
      "n@2": 41,
      "n@3": 25,
      "patient_macro_mse@1": 0.8087465455360484,
      "patient_macro_mse@2": 0.8674317326707144,
      "patient_macro_mse@3": 0.979684130711989,
      "patients@1": 22,
      "patients@2": 16,
      "patients@3": 11
    },
    "rrt/7": {
      "CosSim@1": 0.6315571424506959,
      "CosSim@2": 0.5651479234419218,
      "CosSim@3": 0.5037094950675964,
      "long_mse": 0.9279277981854067,
      "mse@1": 0.7113286803166071,
      "mse@2": 0.8286623922063083,
      "mse@3": 1.027193204164505,
      "n@1": 63,
      "n@2": 41,
      "n@3": 25,
      "patient_macro_mse@1": 0.8139942577165183,
      "patient_macro_mse@2": 0.8830177478957921,
      "patient_macro_mse@3": 1.0322096948370787,
      "patients@1": 22,
      "patients@2": 16,
      "patients@3": 11
    },
    "rrt_ensemble/17": {
      "CosSim@1": 0.6433567772545512,
      "CosSim@2": 0.5982286951890806,
      "CosSim@3": 0.5732024350762367,
      "long_mse": 0.8001314004238059,
      "mse@1": 0.6935858126907122,
      "mse@2": 0.7522507629743437,
      "mse@3": 0.8480120378732682,
      "n@1": 63,
      "n@2": 41,
      "n@3": 25,
      "patient_macro_mse@1": 0.8105895865877919,
      "patient_macro_mse@2": 0.8171775763233504,
      "patient_macro_mse@3": 0.8255228540211014,
      "patients@1": 22,
      "patients@2": 16,
      "patients@3": 11
    },
    "rrt_ensemble/29": {
      "CosSim@1": 0.641390660925517,
      "CosSim@2": 0.5907047510873981,
      "CosSim@3": 0.5624182724952698,
      "long_mse": 0.7878649233899465,
      "mse@1": 0.6727093161335067,
      "mse@2": 0.7406364719315273,
      "mse@3": 0.8350933748483658,
      "n@1": 63,
      "n@2": 41,
      "n@3": 25,
      "patient_macro_mse@1": 0.7903463474277292,
      "patient_macro_mse@2": 0.8054895805350194,
      "patient_macro_mse@3": 0.8082209459759973,
      "patients@1": 22,
      "patients@2": 16,
      "patients@3": 11
    },
    "rrt_ensemble/7": {
      "CosSim@1": 0.6408329039575562,
      "CosSim@2": 0.5892287996120569,
      "CosSim@3": 0.5537027847766877,
      "long_mse": 0.7736814068584907,
      "mse@1": 0.6556995855200858,
      "mse@2": 0.7229918394146896,
      "mse@3": 0.8243709743022919,
      "n@1": 63,
      "n@2": 41,
      "n@3": 25,
      "patient_macro_mse@1": 0.7747646448851534,
      "patient_macro_mse@2": 0.7895782208846261,
      "patient_macro_mse@3": 0.8073206986441757,
      "patients@1": 22,
      "patients@2": 16,
      "patients@3": 11
    }
  },
  "validation": {
    "baseline/17": {
      "CosSim@1": 0.5487768885589415,
      "CosSim@2": 0.4908265949059755,
      "CosSim@3": 0.43812640430405736,
      "long_mse": 0.9490951903181755,
      "mse@1": 0.8449069348073774,
      "mse@2": 0.9729635906525147,
      "mse@3": 0.9252267899838361,
      "n@1": 62,
      "n@2": 39,
      "n@3": 22,
      "patient_macro_mse@1": 0.9578824111948844,
      "patient_macro_mse@2": 1.255202968477034,
      "patient_macro_mse@3": 0.9053137661130339,
      "patients@1": 23,
      "patients@2": 17,
      "patients@3": 9
    },
    "baseline/29": {
      "CosSim@1": 0.5466920263133943,
      "CosSim@2": 0.514928475786478,
      "CosSim@3": 0.48223307847299357,
      "long_mse": 0.956874880635794,
      "mse@1": 0.8959909225663831,
      "mse@2": 0.9803465402279145,
      "mse@3": 0.9334032210436735,
      "n@1": 62,
      "n@2": 39,
      "n@3": 22,
      "patient_macro_mse@1": 1.0154134893654914,
      "patient_macro_mse@2": 1.300287750348741,
      "patient_macro_mse@3": 0.9147790929785482,
      "patients@1": 23,
      "patients@2": 17,
      "patients@3": 9
    },
    "baseline/7": {
      "CosSim@1": 0.5457258773066344,
      "CosSim@2": 0.5082350967881771,
      "CosSim@3": 0.46220348301258957,
      "long_mse": 0.94893486976693,
      "mse@1": 0.8710462580765447,
      "mse@2": 0.9675702827099042,
      "mse@3": 0.9302994568239559,
      "n@1": 62,
      "n@2": 39,
      "n@3": 22,
      "patient_macro_mse@1": 0.9871046183981759,
      "patient_macro_mse@2": 1.2574800307698109,
      "patient_macro_mse@3": 0.9079425478423082,
      "patients@1": 23,
      "patients@2": 17,
      "patients@3": 9
    },
    "ensemble/17": {
      "CosSim@1": 0.5567948159972026,
      "CosSim@2": 0.5269831740894378,
      "CosSim@3": 0.49931112744591455,
      "long_mse": 0.8307965946642113,
      "mse@1": 0.8139950697941165,
      "mse@2": 0.8871982368903283,
      "mse@3": 0.7743949524380944,
      "n@1": 62,
      "n@2": 39,
      "n@3": 22,
      "patient_macro_mse@1": 0.9263552808459256,
      "patient_macro_mse@2": 1.1560346411869806,
      "patient_macro_mse@3": 0.7606330541548906,
      "patients@1": 23,
      "patients@2": 17,
      "patients@3": 9
    },
    "ensemble/29": {
      "CosSim@1": 0.5518504463947348,
      "CosSim@2": 0.5198633634509184,
      "CosSim@3": 0.4826623374088244,
      "long_mse": 0.8356169482176399,
      "mse@1": 0.8152356671710168,
      "mse@2": 0.8907055312242264,
      "mse@3": 0.7805283652110533,
      "n@1": 62,
      "n@2": 39,
      "n@3": 22,
      "patient_macro_mse@1": 0.9308001468147056,
      "patient_macro_mse@2": 1.16168432346746,
      "patient_macro_mse@3": 0.7639220986101364,
      "patients@1": 23,
      "patients@2": 17,
      "patients@3": 9
    },
    "ensemble/7": {
      "CosSim@1": 0.5591847373232726,
      "CosSim@2": 0.5346969270553344,
      "CosSim@3": 0.5104479139501398,
      "long_mse": 0.8165677929848503,
      "mse@1": 0.8103067551649386,
      "mse@2": 0.875001195149544,
      "mse@3": 0.7581343908201564,
      "n@1": 62,
      "n@2": 39,
      "n@3": 22,
      "patient_macro_mse@1": 0.9260072616941255,
      "patient_macro_mse@2": 1.1484649975364116,
      "patient_macro_mse@3": 0.7453309239612685,
      "patients@1": 23,
      "patients@2": 17,
      "patients@3": 9
    },
    "paired_relative_improvement": {
      "baseline_vs_rrt": {
        "mean": 6.3852257712663665,
        "n": 3,
        "per_seed": {
          "17": 6.339861736567229,
          "29": 6.832810834160503,
          "7": 5.983004743071367
        },
        "std": 0.4267153850591602
      },
      "baseline_vs_rrt_ensemble": {
        "mean": 14.299237155091985,
        "n": 3,
        "per_seed": {
          "17": 14.033785073094938,
          "29": 15.004239571112292,
          "7": 13.859686821068726
        },
        "std": 0.616724294385666
      },
      "ensemble_vs_rrt_ensemble": {
        "mean": 1.4531468593039367,
        "n": 3,
        "per_seed": {
          "17": 1.7929037733225528,
          "29": 2.670346396854495,
          "7": -0.1038095922652374
        },
        "std": 1.4179426763659417
      }
    },
    "persistence": {
      "CosSim@1": 0.5501530480030323,
      "CosSim@2": 0.5434200843939414,
      "CosSim@3": 0.5442314351146872,
      "long_mse": 0.889062368847204,
      "mse@1": 0.916118394463293,
      "mse@2": 0.9471973677476248,
      "mse@3": 0.8309273699467833,
      "n@1": 62,
      "n@2": 39,
      "n@3": 22,
      "patient_macro_mse@1": 1.0342471776125224,
      "patient_macro_mse@2": 1.282342699593773,
      "patient_macro_mse@3": 0.8164101654180773,
      "patients@1": 23,
      "patients@2": 17,
      "patients@3": 9
    },
    "rrt/17": {
      "CosSim@1": 0.5471628892926439,
      "CosSim@2": 0.5191895397714316,
      "CosSim@3": 0.48283385925672273,
      "long_mse": 0.8889238675035935,
      "mse@1": 0.8651639862406638,
      "mse@2": 0.9245278116984245,
      "mse@3": 0.8533199233087626,
      "n@1": 62,
      "n@2": 39,
      "n@3": 22,
      "patient_macro_mse@1": 0.9791526040089302,
      "patient_macro_mse@2": 1.2292463580036865,
      "patient_macro_mse@3": 0.8368981028044664,
      "patients@1": 23,
      "patients@2": 17,
      "patients@3": 9
    },
    "rrt/29": {
      "CosSim@1": 0.543162114377464,
      "CosSim@2": 0.5137759630974287,
      "CosSim@3": 0.4819103812968189,
      "long_mse": 0.891493430122351,
      "mse@1": 0.8712896019701035,
      "mse@2": 0.9288758146457183,
      "mse@3": 0.8541110455989838,
      "n@1": 62,
      "n@2": 39,
      "n@3": 22,
      "patient_macro_mse@1": 0.9847799219083094,
      "patient_macro_mse@2": 1.2345328935805489,
      "patient_macro_mse@3": 0.8323506172056552,
      "patients@1": 23,
      "patients@2": 17,
      "patients@3": 9
    },
    "rrt/7": {
      "CosSim@1": 0.5448562267476753,
      "CosSim@2": 0.5214063162939289,
      "CosSim@3": 0.4870418250899423,
      "long_mse": 0.8921600515001165,
      "mse@1": 0.8648980562725375,
      "mse@2": 0.9228269740557059,
      "mse@3": 0.861493128944527,
      "n@1": 62,
      "n@2": 39,
      "n@3": 22,
      "patient_macro_mse@1": 0.980024869405273,
      "patient_macro_mse@2": 1.219601310059136,
      "patient_macro_mse@3": 0.8411613023943372,
      "patients@1": 23,
      "patients@2": 17,
      "patients@3": 9
    },
    "rrt_ensemble/17": {
      "CosSim@1": 0.5527184634739833,
      "CosSim@2": 0.5408739714572827,
      "CosSim@3": 0.5349197560413317,
      "long_mse": 0.8159012111698414,
      "mse@1": 0.8552338228591027,
      "mse@2": 0.8800138258017026,
      "mse@3": 0.7517885965379801,
      "n@1": 62,
      "n@2": 39,
      "n@3": 22,
      "patient_macro_mse@1": 0.9714129562503188,
      "patient_macro_mse@2": 1.1839312486642715,
      "patient_macro_mse@3": 0.741433398867095,
      "patients@1": 23,
      "patients@2": 17,
      "patients@3": 9
    },
    "rrt_ensemble/29": {
      "CosSim@1": 0.5513550926135072,
      "CosSim@2": 0.5371711054124321,
      "CosSim@3": 0.5290081863376227,
      "long_mse": 0.8133030811494046,
      "mse@1": 0.8480816000411587,
      "mse@2": 0.8776774853467941,
      "mse@3": 0.7489286769520153,
      "n@1": 62,
      "n@2": 39,
      "n@3": 22,
      "patient_macro_mse@1": 0.9647094226833702,
      "patient_macro_mse@2": 1.1807122800280065,
      "patient_macro_mse@3": 0.7359991741401177,
      "patients@1": 23,
      "patients@2": 17,
      "patients@3": 9
    },
    "rrt_ensemble/7": {
      "CosSim@1": 0.5493980108670169,
      "CosSim@2": 0.5340067961324866,
      "CosSim@3": 0.5210708030922846,
      "long_mse": 0.8174154686813171,
      "mse@1": 0.8453003136861709,
      "mse@2": 0.8764827927717795,
      "mse@3": 0.7583481445908546,
      "n@1": 62,
      "n@2": 39,
      "n@3": 22,
      "patient_macro_mse@1": 0.9601577515079491,
      "patient_macro_mse@2": 1.1666470393538475,
      "patient_macro_mse@3": 0.7441485019193756,
      "patients@1": 23,
      "patients@2": 17,
      "patients@3": 9
    }
  }
}
```

## Ensemble reliability

```json
{
  "test": {
    "ensemble/17": {
      "H1": {
        "coverage": 0.7936507936507936,
        "high_low_error_ratio": 2.17745210537799,
        "n": 63,
        "ratio_reason": null,
        "retained": 50,
        "risk@100": 0.7159880610212447,
        "risk@80": 0.5421050879359245,
        "spearman": 0.3605030721966206,
        "spearman_reason": null,
        "uncertainty_q90": 0.0918688446283341
      },
      "H2": {
        "coverage": 0.7804878048780488,
        "high_low_error_ratio": 2.528171066924441,
        "n": 41,
        "ratio_reason": null,
        "retained": 32,
        "risk@100": 0.8237566788022112,
        "risk@80": 0.6021643504500389,
        "spearman": 0.3829268292682928,
        "spearman_reason": null,
        "uncertainty_q90": 0.1955438107252121
      },
      "H3": {
        "coverage": 0.8,
        "high_low_error_ratio": 2.8833634807509547,
        "n": 25,
        "ratio_reason": null,
        "retained": 20,
        "risk@100": 0.9405534958839417,
        "risk@80": 0.6820196807384491,
        "spearman": 0.43999999999999995,
        "spearman_reason": null,
        "uncertainty_q90": 0.44265753030777
      }
    },
    "ensemble/29": {
      "H1": {
        "coverage": 0.7936507936507936,
        "high_low_error_ratio": 2.420336392638935,
        "n": 63,
        "ratio_reason": null,
        "retained": 50,
        "risk@100": 0.7260130830700435,
        "risk@80": 0.5465304234623909,
        "spearman": 0.42789938556067597,
        "spearman_reason": null,
        "uncertainty_q90": 0.09070146083831788
      },
      "H2": {
        "coverage": 0.7804878048780488,
        "high_low_error_ratio": 2.5505065204822333,
        "n": 41,
        "ratio_reason": null,
        "retained": 32,
        "risk@100": 0.8253457415394667,
        "risk@80": 0.5595872150734067,
        "spearman": 0.4942508710801394,
        "spearman_reason": null,
        "uncertainty_q90": 0.16297496855258942
      },
      "H3": {
        "coverage": 0.8,
        "high_low_error_ratio": 2.6132826988835856,
        "n": 25,
        "ratio_reason": null,
        "retained": 20,
        "risk@100": 0.9387907779216766,
        "risk@80": 0.6399533674120903,
        "spearman": 0.4938461538461539,
        "spearman_reason": null,
        "uncertainty_q90": 0.4475226283073428
      }
    },
    "ensemble/7": {
      "H1": {
        "coverage": 0.7936507936507936,
        "high_low_error_ratio": 2.2296891003165857,
        "n": 63,
        "ratio_reason": null,
        "retained": 50,
        "risk@100": 0.7095490242280658,
        "risk@80": 0.5288130751252175,
        "spearman": 0.32301267281105994,
        "spearman_reason": null,
        "uncertainty_q90": 0.12430320382118226
      },
      "H2": {
        "coverage": 0.7804878048780488,
        "high_low_error_ratio": 2.3075056184267493,
        "n": 41,
        "ratio_reason": null,
        "retained": 32,
        "risk@100": 0.8065680945064964,
        "risk@80": 0.580314505379647,
        "spearman": 0.3506968641114983,
        "spearman_reason": null,
        "uncertainty_q90": 0.22633248567581177
      },
      "H3": {
        "coverage": 0.8,
        "high_low_error_ratio": 2.5030880727975497,
        "n": 25,
        "ratio_reason": null,
        "retained": 20,
        "risk@100": 0.923191260099411,
        "risk@80": 0.6723372399806976,
        "spearman": 0.37538461538461537,
        "spearman_reason": null,
        "uncertainty_q90": 0.5135136127471926
      }
    },
    "rrt_ensemble/17": {
      "H1": {
        "coverage": 0.7936507936507936,
        "high_low_error_ratio": 2.33387969173723,
        "n": 63,
        "ratio_reason": null,
        "retained": 50,
        "risk@100": 0.6935858126907122,
        "risk@80": 0.5178969456255436,
        "spearman": 0.282258064516129,
        "spearman_reason": null,
        "uncertainty_q90": 0.033955156058073054
      },
      "H2": {
        "coverage": 0.7804878048780488,
        "high_low_error_ratio": 2.2472050078811128,
        "n": 41,
        "ratio_reason": null,
        "retained": 32,
        "risk@100": 0.7522507629743437,
        "risk@80": 0.5668803378939629,
        "spearman": 0.23379790940766557,
        "spearman_reason": null,
        "uncertainty_q90": 0.09216687083244324
      },
      "H3": {
        "coverage": 0.8,
        "high_low_error_ratio": 1.8837466585869849,
        "n": 25,
        "ratio_reason": null,
        "retained": 20,
        "risk@100": 0.8480120378732682,
        "risk@80": 0.6690606690943242,
        "spearman": 0.16923076923076924,
        "spearman_reason": null,
        "uncertainty_q90": 0.19353390038013468
      }
    },
    "rrt_ensemble/29": {
      "H1": {
        "coverage": 0.7936507936507936,
        "high_low_error_ratio": 2.2143722998951225,
        "n": 63,
        "ratio_reason": null,
        "retained": 50,
        "risk@100": 0.6727093161335067,
        "risk@80": 0.5190882153809071,
        "spearman": 0.28446620583717364,
        "spearman_reason": null,
        "uncertainty_q90": 0.042059558629989634
      },
      "H2": {
        "coverage": 0.7804878048780488,
        "high_low_error_ratio": 2.26598786048265,
        "n": 41,
        "ratio_reason": null,
        "retained": 32,
        "risk@100": 0.7406364719315273,
        "risk@80": 0.5766716604121029,
        "spearman": 0.28815331010452966,
        "spearman_reason": null,
        "uncertainty_q90": 0.12137839198112488
      },
      "H3": {
        "coverage": 0.8,
        "high_low_error_ratio": 1.7298987568282362,
        "n": 25,
        "ratio_reason": null,
        "retained": 20,
        "risk@100": 0.8350933748483658,
        "risk@80": 0.68138183131814,
        "spearman": 0.11923076923076922,
        "spearman_reason": null,
        "uncertainty_q90": 0.23327567577362074
      }
    },
    "rrt_ensemble/7": {
      "H1": {
        "coverage": 0.7936507936507936,
        "high_low_error_ratio": 2.3271504545273594,
        "n": 63,
        "ratio_reason": null,
        "retained": 50,
        "risk@100": 0.6556995855200858,
        "risk@80": 0.5194434873759747,
        "spearman": 0.30870775729646693,
        "spearman_reason": null,
        "uncertainty_q90": 0.03856203705072403
      },
      "H2": {
        "coverage": 0.7804878048780488,
        "high_low_error_ratio": 2.0718459042843325,
        "n": 41,
        "ratio_reason": null,
        "retained": 32,
        "risk@100": 0.7229918394146896,
        "risk@80": 0.5787349445745349,
        "spearman": 0.28658536585365857,
        "spearman_reason": null,
        "uncertainty_q90": 0.10126618295907974
      },
      "H3": {
        "coverage": 0.8,
        "high_low_error_ratio": 1.578954894962951,
        "n": 25,
        "ratio_reason": null,
        "retained": 20,
        "risk@100": 0.8243709743022919,
        "risk@80": 0.6881597712635994,
        "spearman": 0.14307692307692307,
        "spearman_reason": null,
        "uncertainty_q90": 0.1984119713306428
      }
    }
  },
  "validation": {
    "ensemble/17": {
      "H1": {
        "coverage": 0.7903225806451613,
        "high_low_error_ratio": 1.1614586548478352,
        "n": 62,
        "ratio_reason": null,
        "retained": 49,
        "risk@100": 0.8139950697941165,
        "risk@80": 0.7870471039596869,
        "spearman": 0.15781521492785375,
        "spearman_reason": null,
        "uncertainty_q90": 0.0744799830019474
      },
      "H2": {
        "coverage": 0.7948717948717948,
        "high_low_error_ratio": 1.3450875742427724,
        "n": 39,
        "ratio_reason": null,
        "retained": 31,
        "risk@100": 0.8871982368903283,
        "risk@80": 0.8493377003938921,
        "spearman": 0.18805668016194332,
        "spearman_reason": null,
        "uncertainty_q90": 0.16259605586528786
      },
      "H3": {
        "coverage": 0.7727272727272727,
        "high_low_error_ratio": 0.727601609187599,
        "n": 22,
        "ratio_reason": null,
        "retained": 17,
        "risk@100": 0.7743949524380944,
        "risk@80": 0.7557355340789346,
        "spearman": 0.1993224167137211,
        "spearman_reason": null,
        "uncertainty_q90": 0.25098755508661275
      }
    },
    "ensemble/29": {
      "H1": {
        "coverage": 0.7903225806451613,
        "high_low_error_ratio": 1.407433884124649,
        "n": 62,
        "ratio_reason": null,
        "retained": 49,
        "risk@100": 0.8152356671710168,
        "risk@80": 0.7964062721145396,
        "spearman": 0.16406033592707311,
        "spearman_reason": null,
        "uncertainty_q90": 0.07813680842518805
      },
      "H2": {
        "coverage": 0.7948717948717948,
        "high_low_error_ratio": 1.5413215273146217,
        "n": 39,
        "ratio_reason": null,
        "retained": 31,
        "risk@100": 0.8907055312242264,
        "risk@80": 0.8552438068774438,
        "spearman": 0.19068825910931173,
        "spearman_reason": null,
        "uncertainty_q90": 0.1595236659049988
      },
      "H3": {
        "coverage": 0.7727272727272727,
        "high_low_error_ratio": 0.7335676786711404,
        "n": 22,
        "ratio_reason": null,
        "retained": 17,
        "risk@100": 0.7805283652110533,
        "risk@80": 0.7659185125547296,
        "spearman": 0.1959345002823264,
        "spearman_reason": null,
        "uncertainty_q90": 0.25493616014719017
      }
    },
    "ensemble/7": {
      "H1": {
        "coverage": 0.7903225806451613,
        "high_low_error_ratio": 1.1772844317358238,
        "n": 62,
        "ratio_reason": null,
        "retained": 49,
        "risk@100": 0.8103067551649386,
        "risk@80": 0.7814491940092068,
        "spearman": 0.1862708065775226,
        "spearman_reason": null,
        "uncertainty_q90": 0.1052253223955631
      },
      "H2": {
        "coverage": 0.7948717948717948,
        "high_low_error_ratio": 1.487276517505862,
        "n": 39,
        "ratio_reason": null,
        "retained": 31,
        "risk@100": 0.875001195149544,
        "risk@80": 0.8447358694768721,
        "spearman": 0.16761133603238867,
        "spearman_reason": null,
        "uncertainty_q90": 0.17781503498554246
      },
      "H3": {
        "coverage": 0.7727272727272727,
        "high_low_error_ratio": 0.6821002945753168,
        "n": 22,
        "ratio_reason": null,
        "retained": 17,
        "risk@100": 0.7581343908201564,
        "risk@80": 0.7442172993631924,
        "spearman": 0.09203839638622249,
        "spearman_reason": null,
        "uncertainty_q90": 0.25582257807254793
      }
    },
    "rrt_ensemble/17": {
      "H1": {
        "coverage": 0.7903225806451613,
        "high_low_error_ratio": 1.2477728828586336,
        "n": 62,
        "ratio_reason": null,
        "retained": 49,
        "risk@100": 0.8552338228591027,
        "risk@80": 0.8432594191054908,
        "spearman": 0.10060184835436024,
        "spearman_reason": null,
        "uncertainty_q90": 0.03521642349660396
      },
      "H2": {
        "coverage": 0.7948717948717948,
        "high_low_error_ratio": 0.9284823268160216,
        "n": 39,
        "ratio_reason": null,
        "retained": 31,
        "risk@100": 0.8800138258017026,
        "risk@80": 0.8558083960125523,
        "spearman": 0.053238866396761134,
        "spearman_reason": null,
        "uncertainty_q90": 0.09682016968727118
      },
      "H3": {
        "coverage": 0.7727272727272727,
        "high_low_error_ratio": 1.8501658677041553,
        "n": 22,
        "ratio_reason": null,
        "retained": 17,
        "risk@100": 0.7517885965379801,
        "risk@80": 0.724201483761563,
        "spearman": 0.16544325239977417,
        "spearman_reason": null,
        "uncertainty_q90": 0.15158531218767174
      }
    },
    "rrt_ensemble/29": {
      "H1": {
        "coverage": 0.7903225806451613,
        "high_low_error_ratio": 1.2422019231731245,
        "n": 62,
        "ratio_reason": null,
        "retained": 49,
        "risk@100": 0.8480816000411587,
        "risk@80": 0.835528949085547,
        "spearman": 0.06237566417365466,
        "spearman_reason": null,
        "uncertainty_q90": 0.038016024976968765
      },
      "H2": {
        "coverage": 0.7948717948717948,
        "high_low_error_ratio": 0.9358523807736615,
        "n": 39,
        "ratio_reason": null,
        "retained": 31,
        "risk@100": 0.8776774853467941,
        "risk@80": 0.8429117303702139,
        "spearman": 0.08380566801619434,
        "spearman_reason": null,
        "uncertainty_q90": 0.10915862619876869
      },
      "H3": {
        "coverage": 0.7727272727272727,
        "high_low_error_ratio": 1.5850062222451669,
        "n": 22,
        "ratio_reason": null,
        "retained": 17,
        "risk@100": 0.7489286769520153,
        "risk@80": 0.713591621202581,
        "spearman": 0.1575381140598532,
        "spearman_reason": null,
        "uncertainty_q90": 0.1412520080804825
      }
    },
    "rrt_ensemble/7": {
      "H1": {
        "coverage": 0.7903225806451613,
        "high_low_error_ratio": 1.0211462093929484,
        "n": 62,
        "ratio_reason": null,
        "retained": 49,
        "risk@100": 0.8453003136861709,
        "risk@80": 0.8314519402932148,
        "spearman": 0.04192792928911385,
        "spearman_reason": null,
        "uncertainty_q90": 0.040851409733295437
      },
      "H2": {
        "coverage": 0.7948717948717948,
        "high_low_error_ratio": 0.9884641751964816,
        "n": 39,
        "ratio_reason": null,
        "retained": 31,
        "risk@100": 0.8764827927717795,
        "risk@80": 0.8349080090561221,
        "spearman": 0.09028340080971661,
        "spearman_reason": null,
        "uncertainty_q90": 0.11534127742052079
      },
      "H3": {
        "coverage": 0.7727272727272727,
        "high_low_error_ratio": 0.800610538176918,
        "n": 22,
        "ratio_reason": null,
        "retained": 17,
        "risk@100": 0.7583481445908546,
        "risk@80": 0.7200779415228787,
        "spearman": 0.1518915866741954,
        "spearman_reason": null,
        "uncertainty_q90": 0.15421976000070575
      }
    }
  }
}
```

## Outcome — observational landmark prediction

```json
{
  "test": {
    "O0/17": {
      "all_landmarks_supplemental": {
        "available": true,
        "c_index": {
          "comparable_pairs": 719,
          "reason": null,
          "value": 0.7837273991655076
        },
        "censored": 27,
        "events": 27,
        "ipcw_brier": {
          "n": 54,
          "reason": null,
          "value": 0.1980260914939162
        },
        "landmark": "all_eligible_supplemental",
        "n": 54,
        "observations": 54,
        "patients": 18,
        "repeated_observations": 36,
        "risk_definition": "1-S(365)",
        "survival_nll": 3.464960536643587
      },
      "primary": {
        "available": true,
        "c_index": {
          "comparable_pairs": 64,
          "reason": null,
          "value": 0.84375
        },
        "censored": 11,
        "events": 7,
        "ipcw_brier": {
          "n": 18,
          "reason": null,
          "value": 0.13994977384314364
        },
        "landmark": "first_eligible",
        "n": 18,
        "observations": 18,
        "patients": 18,
        "repeated_observations": 0,
        "risk_definition": "1-S(365)",
        "survival_nll": 2.7953383563500314
      }
    },
    "O0/29": {
      "all_landmarks_supplemental": {
        "available": true,
        "c_index": {
          "comparable_pairs": 719,
          "reason": null,
          "value": 0.7837273991655076
        },
        "censored": 27,
        "events": 27,
        "ipcw_brier": {
          "n": 54,
          "reason": null,
          "value": 0.19841342853574062
        },
        "landmark": "all_eligible_supplemental",
        "n": 54,
        "observations": 54,
        "patients": 18,
        "repeated_observations": 36,
        "risk_definition": "1-S(365)",
        "survival_nll": 3.444561841363018
      },
      "primary": {
        "available": true,
        "c_index": {
          "comparable_pairs": 64,
          "reason": null,
          "value": 0.8125
        },
        "censored": 11,
        "events": 7,
        "ipcw_brier": {
          "n": 18,
          "reason": null,
          "value": 0.13292917921307174
        },
        "landmark": "first_eligible",
        "n": 18,
        "observations": 18,
        "patients": 18,
        "repeated_observations": 0,
        "risk_definition": "1-S(365)",
        "survival_nll": 2.7602121245322957
      }
    },
    "O0/7": {
      "all_landmarks_supplemental": {
        "available": true,
        "c_index": {
          "comparable_pairs": 719,
          "reason": null,
          "value": 0.7698191933240612
        },
        "censored": 27,
        "events": 27,
        "ipcw_brier": {
          "n": 54,
          "reason": null,
          "value": 0.20727562069787228
        },
        "landmark": "all_eligible_supplemental",
        "n": 54,
        "observations": 54,
        "patients": 18,
        "repeated_observations": 36,
        "risk_definition": "1-S(365)",
        "survival_nll": 3.4323577121486544
      },
      "primary": {
        "available": true,
        "c_index": {
          "comparable_pairs": 64,
          "reason": null,
          "value": 0.84375
        },
        "censored": 11,
        "events": 7,
        "ipcw_brier": {
          "n": 18,
          "reason": null,
          "value": 0.13179584538531997
        },
        "landmark": "first_eligible",
        "n": 18,
        "observations": 18,
        "patients": 18,
        "repeated_observations": 0,
        "risk_definition": "1-S(365)",
        "survival_nll": 2.7901717838831246
      }
    },
    "O1/17": {
      "all_landmarks_supplemental": {
        "available": true,
        "c_index": {
          "comparable_pairs": 719,
          "reason": null,
          "value": 0.7357440890125174
        },
        "censored": 27,
        "events": 27,
        "ipcw_brier": {
          "n": 54,
          "reason": null,
          "value": 0.19912209723503624
        },
        "landmark": "all_eligible_supplemental",
        "n": 54,
        "observations": 54,
        "patients": 18,
        "repeated_observations": 36,
        "risk_definition": "1-S(365)",
        "survival_nll": 3.7369786563708827
      },
      "primary": {
        "available": true,
        "c_index": {
          "comparable_pairs": 64,
          "reason": null,
          "value": 0.609375
        },
        "censored": 11,
        "events": 7,
        "ipcw_brier": {
          "n": 18,
          "reason": null,
          "value": 0.2479026085132654
        },
        "landmark": "first_eligible",
        "n": 18,
        "observations": 18,
        "patients": 18,
        "repeated_observations": 0,
        "risk_definition": "1-S(365)",
        "survival_nll": 3.030543219505085
      }
    },
    "O1/29": {
      "all_landmarks_supplemental": {
        "available": true,
        "c_index": {
          "comparable_pairs": 719,
          "reason": null,
          "value": 0.6801112656467315
        },
        "censored": 27,
        "events": 27,
        "ipcw_brier": {
          "n": 54,
          "reason": null,
          "value": 0.2222664707854742
        },
        "landmark": "all_eligible_supplemental",
        "n": 54,
        "observations": 54,
        "patients": 18,
        "repeated_observations": 36,
        "risk_definition": "1-S(365)",
        "survival_nll": 3.770140788966307
      },
      "primary": {
        "available": true,
        "c_index": {
          "comparable_pairs": 64,
          "reason": null,
          "value": 0.515625
        },
        "censored": 11,
        "events": 7,
        "ipcw_brier": {
          "n": 18,
          "reason": null,
          "value": 0.2889654453673713
        },
        "landmark": "first_eligible",
        "n": 18,
        "observations": 18,
        "patients": 18,
        "repeated_observations": 0,
        "risk_definition": "1-S(365)",
        "survival_nll": 3.134565869346261
      }
    },
    "O1/7": {
      "all_landmarks_supplemental": {
        "available": true,
        "c_index": {
          "comparable_pairs": 719,
          "reason": null,
          "value": 0.6648122392211405
        },
        "censored": 27,
        "events": 27,
        "ipcw_brier": {
          "n": 54,
          "reason": null,
          "value": 0.22653045068575217
        },
        "landmark": "all_eligible_supplemental",
        "n": 54,
        "observations": 54,
        "patients": 18,
        "repeated_observations": 36,
        "risk_definition": "1-S(365)",
        "survival_nll": 3.7522298178670033
      },
      "primary": {
        "available": true,
        "c_index": {
          "comparable_pairs": 64,
          "reason": null,
          "value": 0.53125
        },
        "censored": 11,
        "events": 7,
        "ipcw_brier": {
          "n": 18,
          "reason": null,
          "value": 0.2932941602881104
        },
        "landmark": "first_eligible",
        "n": 18,
        "observations": 18,
        "patients": 18,
        "repeated_observations": 0,
        "risk_definition": "1-S(365)",
        "survival_nll": 3.132280502054426
      }
    },
    "O2-baseline/17/H1": {
      "available": true,
      "c_index": {
        "comparable_pairs": 35,
        "reason": null,
        "value": 0.6571428571428571
      },
      "censored": 9,
      "events": 5,
      "horizon": 1,
      "ipcw_brier": {
        "n": 14,
        "reason": null,
        "value": 0.2311188558196176
      },
      "landmark": "first_eligible_target_per_patient",
      "n": 14,
      "observations": 14,
      "patients": 14,
      "risk_definition": "1-S(365)",
      "survival_nll": 2.9920703614396706
    },
    "O2-baseline/17/H2": {
      "available": true,
      "c_index": {
        "comparable_pairs": 15,
        "reason": null,
        "value": 0.6
      },
      "censored": 3,
      "events": 5,
      "horizon": 2,
      "ipcw_brier": {
        "n": 8,
        "reason": null,
        "value": 0.2590401368093659
      },
      "landmark": "first_eligible_target_per_patient",
      "n": 8,
      "observations": 8,
      "patients": 8,
      "risk_definition": "1-S(365)",
      "survival_nll": 4.627451751381159
    },
    "O2-baseline/17/H3": {
      "available": true,
      "c_index": {
        "comparable_pairs": 11,
        "reason": null,
        "value": 0.7272727272727273
      },
      "censored": 3,
      "events": 4,
      "horizon": 3,
      "ipcw_brier": {
        "n": 7,
        "reason": null,
        "value": 0.27371444108042375
      },
      "landmark": "first_eligible_target_per_patient",
      "n": 7,
      "observations": 7,
      "patients": 7,
      "risk_definition": "1-S(365)",
      "survival_nll": 4.127518258988857
    },
    "O2-baseline/29/H1": {
      "available": true,
      "c_index": {
        "comparable_pairs": 35,
        "reason": null,
        "value": 0.45714285714285713
      },
      "censored": 9,
      "events": 5,
      "horizon": 1,
      "ipcw_brier": {
        "n": 14,
        "reason": null,
        "value": 0.29263623399361904
      },
      "landmark": "first_eligible_target_per_patient",
      "n": 14,
      "observations": 14,
      "patients": 14,
      "risk_definition": "1-S(365)",
      "survival_nll": 3.134589772937553
    },
    "O2-baseline/29/H2": {
      "available": true,
      "c_index": {
        "comparable_pairs": 15,
        "reason": null,
        "value": 0.4
      },
      "censored": 3,
      "events": 5,
      "horizon": 2,
      "ipcw_brier": {
        "n": 8,
        "reason": null,
        "value": 0.32352952442306626
      },
      "landmark": "first_eligible_target_per_patient",
      "n": 8,
      "observations": 8,
      "patients": 8,
      "risk_definition": "1-S(365)",
      "survival_nll": 4.751618012785912
    },
    "O2-baseline/29/H3": {
      "available": true,
      "c_index": {
        "comparable_pairs": 11,
        "reason": null,
        "value": 0.45454545454545453
      },
      "censored": 3,
      "events": 4,
      "horizon": 3,
      "ipcw_brier": {
        "n": 7,
        "reason": null,
        "value": 0.32674180914433676
      },
      "landmark": "first_eligible_target_per_patient",
      "n": 7,
      "observations": 7,
      "patients": 7,
      "risk_definition": "1-S(365)",
      "survival_nll": 4.206705226429871
    },
    "O2-baseline/7/H1": {
      "available": true,
      "c_index": {
        "comparable_pairs": 35,
        "reason": null,
        "value": 0.45714285714285713
      },
      "censored": 9,
      "events": 5,
      "horizon": 1,
      "ipcw_brier": {
        "n": 14,
        "reason": null,
        "value": 0.27552390583520675
      },
      "landmark": "first_eligible_target_per_patient",
      "n": 14,
      "observations": 14,
      "patients": 14,
      "risk_definition": "1-S(365)",
      "survival_nll": 3.0742483117750714
    },
    "O2-baseline/7/H2": {
      "available": true,
      "c_index": {
        "comparable_pairs": 15,
        "reason": null,
        "value": 0.4
      },
      "censored": 3,
      "events": 5,
      "horizon": 2,
      "ipcw_brier": {
        "n": 8,
        "reason": null,
        "value": 0.2951319562558256
      },
      "landmark": "first_eligible_target_per_patient",
      "n": 8,
      "observations": 8,
      "patients": 8,
      "risk_definition": "1-S(365)",
      "survival_nll": 4.744467619806528
    },
    "O2-baseline/7/H3": {
      "available": true,
      "c_index": {
        "comparable_pairs": 11,
        "reason": null,
        "value": 0.5454545454545454
      },
      "censored": 3,
      "events": 4,
      "horizon": 3,
      "ipcw_brier": {
        "n": 7,
        "reason": null,
        "value": 0.28261537093819644
      },
      "landmark": "first_eligible_target_per_patient",
      "n": 7,
      "observations": 7,
      "patients": 7,
      "risk_definition": "1-S(365)",
      "survival_nll": 4.093364609139306
    },
    "O2-ensemble/17/H1": {
      "available": true,
      "c_index": {
        "comparable_pairs": 35,
        "reason": null,
        "value": 0.6571428571428571
      },
      "censored": 9,
      "events": 5,
      "horizon": 1,
      "ipcw_brier": {
        "n": 14,
        "reason": null,
        "value": 0.2411402599224489
      },
      "landmark": "first_eligible_target_per_patient",
      "n": 14,
      "observations": 14,
      "patients": 14,
      "risk_definition": "1-S(365)",
      "survival_nll": 3.0178038020219122
    },
    "O2-ensemble/17/H2": {
      "available": true,
      "c_index": {
        "comparable_pairs": 15,
        "reason": null,
        "value": 0.6
      },
      "censored": 3,
      "events": 5,
      "horizon": 2,
      "ipcw_brier": {
        "n": 8,
        "reason": null,
        "value": 0.2643600817939611
      },
      "landmark": "first_eligible_target_per_patient",
      "n": 8,
      "observations": 8,
      "patients": 8,
      "risk_definition": "1-S(365)",
      "survival_nll": 4.651397586800158
    },
    "O2-ensemble/17/H3": {
      "available": true,
      "c_index": {
        "comparable_pairs": 11,
        "reason": null,
        "value": 0.8181818181818182
      },
      "censored": 3,
      "events": 4,
      "horizon": 3,
      "ipcw_brier": {
        "n": 7,
        "reason": null,
        "value": 0.2634333333574
      },
      "landmark": "first_eligible_target_per_patient",
      "n": 7,
      "observations": 7,
      "patients": 7,
      "risk_definition": "1-S(365)",
      "survival_nll": 4.085030052278723
    },
    "O2-ensemble/29/H1": {
      "available": true,
      "c_index": {
        "comparable_pairs": 35,
        "reason": null,
        "value": 0.5142857142857142
      },
      "censored": 9,
      "events": 5,
      "horizon": 1,
      "ipcw_brier": {
        "n": 14,
        "reason": null,
        "value": 0.26265422656859705
      },
      "landmark": "first_eligible_target_per_patient",
      "n": 14,
      "observations": 14,
      "patients": 14,
      "risk_definition": "1-S(365)",
      "survival_nll": 3.1119915981377875
    },
    "O2-ensemble/29/H2": {
      "available": true,
      "c_index": {
        "comparable_pairs": 15,
        "reason": null,
        "value": 0.4666666666666667
      },
      "censored": 3,
      "events": 5,
      "horizon": 2,
      "ipcw_brier": {
        "n": 8,
        "reason": null,
        "value": 0.2734512523484633
      },
      "landmark": "first_eligible_target_per_patient",
      "n": 8,
      "observations": 8,
      "patients": 8,
      "risk_definition": "1-S(365)",
      "survival_nll": 4.665933884680271
    },
    "O2-ensemble/29/H3": {
      "available": true,
      "c_index": {
        "comparable_pairs": 11,
        "reason": null,
        "value": 0.6363636363636364
      },
      "censored": 3,
      "events": 4,
      "horizon": 3,
      "ipcw_brier": {
        "n": 7,
        "reason": null,
        "value": 0.28658813632333857
      },
      "landmark": "first_eligible_target_per_patient",
      "n": 7,
      "observations": 7,
      "patients": 7,
      "risk_definition": "1-S(365)",
      "survival_nll": 4.05940051057509
    },
    "O2-ensemble/7/H1": {
      "available": true,
      "c_index": {
        "comparable_pairs": 35,
        "reason": null,
        "value": 0.5142857142857142
      },
      "censored": 9,
      "events": 5,
      "horizon": 1,
      "ipcw_brier": {
        "n": 14,
        "reason": null,
        "value": 0.2631561453976978
      },
      "landmark": "first_eligible_target_per_patient",
      "n": 14,
      "observations": 14,
      "patients": 14,
      "risk_definition": "1-S(365)",
      "survival_nll": 3.0611573872821674
    },
    "O2-ensemble/7/H2": {
      "available": true,
      "c_index": {
        "comparable_pairs": 15,
        "reason": null,
        "value": 0.4666666666666667
      },
      "censored": 3,
      "events": 5,
      "horizon": 2,
      "ipcw_brier": {
        "n": 8,
        "reason": null,
        "value": 0.2890602664801275
      },
      "landmark": "first_eligible_target_per_patient",
      "n": 8,
      "observations": 8,
      "patients": 8,
      "risk_definition": "1-S(365)",
      "survival_nll": 4.720411426387727
    },
    "O2-ensemble/7/H3": {
      "available": true,
      "c_index": {
        "comparable_pairs": 11,
        "reason": null,
        "value": 0.45454545454545453
      },
      "censored": 3,
      "events": 4,
      "horizon": 3,
      "ipcw_brier": {
        "n": 7,
        "reason": null,
        "value": 0.292120316780915
      },
      "landmark": "first_eligible_target_per_patient",
      "n": 7,
      "observations": 7,
      "patients": 7,
      "risk_definition": "1-S(365)",
      "survival_nll": 4.13449846633843
    },
    "O2-rrt/17/H1": {
      "available": true,
      "c_index": {
        "comparable_pairs": 35,
        "reason": null,
        "value": 0.6285714285714286
      },
      "censored": 9,
      "events": 5,
      "horizon": 1,
      "ipcw_brier": {
        "n": 14,
        "reason": null,
        "value": 0.22373581382157642
      },
      "landmark": "first_eligible_target_per_patient",
      "n": 14,
      "observations": 14,
      "patients": 14,
      "risk_definition": "1-S(365)",
      "survival_nll": 2.979336863649743
    },
    "O2-rrt/17/H2": {
      "available": true,
      "c_index": {
        "comparable_pairs": 15,
        "reason": null,
        "value": 0.6
      },
      "censored": 3,
      "events": 5,
      "horizon": 2,
      "ipcw_brier": {
        "n": 8,
        "reason": null,
        "value": 0.2665572890819582
      },
      "landmark": "first_eligible_target_per_patient",
      "n": 8,
      "observations": 8,
      "patients": 8,
      "risk_definition": "1-S(365)",
      "survival_nll": 4.64039480406791
    },
    "O2-rrt/17/H3": {
      "available": true,
      "c_index": {
        "comparable_pairs": 11,
        "reason": null,
        "value": 0.6363636363636364
      },
      "censored": 3,
      "events": 4,
      "horizon": 3,
      "ipcw_brier": {
        "n": 7,
        "reason": null,
        "value": 0.2654722765946091
      },
      "landmark": "first_eligible_target_per_patient",
      "n": 7,
      "observations": 7,
      "patients": 7,
      "risk_definition": "1-S(365)",
      "survival_nll": 4.1286458522081375
    },
    "O2-rrt/29/H1": {
      "available": true,
      "c_index": {
        "comparable_pairs": 35,
        "reason": null,
        "value": 0.5428571428571428
      },
      "censored": 9,
      "events": 5,
      "horizon": 1,
      "ipcw_brier": {
        "n": 14,
        "reason": null,
        "value": 0.24868900191019314
      },
      "landmark": "first_eligible_target_per_patient",
      "n": 14,
      "observations": 14,
      "patients": 14,
      "risk_definition": "1-S(365)",
      "survival_nll": 3.0997296076800143
    },
    "O2-rrt/29/H2": {
      "available": true,
      "c_index": {
        "comparable_pairs": 15,
        "reason": null,
        "value": 0.4
      },
      "censored": 3,
      "events": 5,
      "horizon": 2,
      "ipcw_brier": {
        "n": 8,
        "reason": null,
        "value": 0.27062455989395817
      },
      "landmark": "first_eligible_target_per_patient",
      "n": 8,
      "observations": 8,
      "patients": 8,
      "risk_definition": "1-S(365)",
      "survival_nll": 4.651144144125283
    },
    "O2-rrt/29/H3": {
      "available": true,
      "c_index": {
        "comparable_pairs": 11,
        "reason": null,
        "value": 0.45454545454545453
      },
      "censored": 3,
      "events": 4,
      "horizon": 3,
      "ipcw_brier": {
        "n": 7,
        "reason": null,
        "value": 0.30142899370434856
      },
      "landmark": "first_eligible_target_per_patient",
      "n": 7,
      "observations": 7,
      "patients": 7,
      "risk_definition": "1-S(365)",
      "survival_nll": 4.1013713883502145
    },
    "O2-rrt/7/H1": {
      "available": true,
      "c_index": {
        "comparable_pairs": 35,
        "reason": null,
        "value": 0.4857142857142857
      },
      "censored": 9,
      "events": 5,
      "horizon": 1,
      "ipcw_brier": {
        "n": 14,
        "reason": null,
        "value": 0.2576779067439117
      },
      "landmark": "first_eligible_target_per_patient",
      "n": 14,
      "observations": 14,
      "patients": 14,
      "risk_definition": "1-S(365)",
      "survival_nll": 3.064067408176405
    },
    "O2-rrt/7/H2": {
      "available": true,
      "c_index": {
        "comparable_pairs": 15,
        "reason": null,
        "value": 0.4
      },
      "censored": 3,
      "events": 5,
      "horizon": 2,
      "ipcw_brier": {
        "n": 8,
        "reason": null,
        "value": 0.2786119740420516
      },
      "landmark": "first_eligible_target_per_patient",
      "n": 8,
      "observations": 8,
      "patients": 8,
      "risk_definition": "1-S(365)",
      "survival_nll": 4.732129035517573
    },
    "O2-rrt/7/H3": {
      "available": true,
      "c_index": {
        "comparable_pairs": 11,
        "reason": null,
        "value": 0.5454545454545454
      },
      "censored": 3,
      "events": 4,
      "horizon": 3,
      "ipcw_brier": {
        "n": 7,
        "reason": null,
        "value": 0.26326271134927287
      },
      "landmark": "first_eligible_target_per_patient",
      "n": 7,
      "observations": 7,
      "patients": 7,
      "risk_definition": "1-S(365)",
      "survival_nll": 4.111585338200841
    },
    "O2-rrt_ensemble/17/H1": {
      "available": true,
      "c_index": {
        "comparable_pairs": 35,
        "reason": null,
        "value": 0.6
      },
      "censored": 9,
      "events": 5,
      "horizon": 1,
      "ipcw_brier": {
        "n": 14,
        "reason": null,
        "value": 0.23949577946498346
      },
      "landmark": "first_eligible_target_per_patient",
      "n": 14,
      "observations": 14,
      "patients": 14,
      "risk_definition": "1-S(365)",
      "survival_nll": 2.99795945041946
    },
    "O2-rrt_ensemble/17/H2": {
      "available": true,
      "c_index": {
        "comparable_pairs": 15,
        "reason": null,
        "value": 0.5333333333333333
      },
      "censored": 3,
      "events": 5,
      "horizon": 2,
      "ipcw_brier": {
        "n": 8,
        "reason": null,
        "value": 0.27361105229643934
      },
      "landmark": "first_eligible_target_per_patient",
      "n": 8,
      "observations": 8,
      "patients": 8,
      "risk_definition": "1-S(365)",
      "survival_nll": 4.68032184522599
    },
    "O2-rrt_ensemble/17/H3": {
      "available": true,
      "c_index": {
        "comparable_pairs": 11,
        "reason": null,
        "value": 0.6363636363636364
      },
      "censored": 3,
      "events": 4,
      "horizon": 3,
      "ipcw_brier": {
        "n": 7,
        "reason": null,
        "value": 0.2610461850982548
      },
      "landmark": "first_eligible_target_per_patient",
      "n": 7,
      "observations": 7,
      "patients": 7,
      "risk_definition": "1-S(365)",
      "survival_nll": 4.134223151419844
    },
    "O2-rrt_ensemble/29/H1": {
      "available": true,
      "c_index": {
        "comparable_pairs": 35,
        "reason": null,
        "value": 0.5142857142857142
      },
      "censored": 9,
      "events": 5,
      "horizon": 1,
      "ipcw_brier": {
        "n": 14,
        "reason": null,
        "value": 0.24975787748649164
      },
      "landmark": "first_eligible_target_per_patient",
      "n": 14,
      "observations": 14,
      "patients": 14,
      "risk_definition": "1-S(365)",
      "survival_nll": 3.091623163915106
    },
    "O2-rrt_ensemble/29/H2": {
      "available": true,
      "c_index": {
        "comparable_pairs": 15,
        "reason": null,
        "value": 0.4666666666666667
      },
      "censored": 3,
      "events": 5,
      "horizon": 2,
      "ipcw_brier": {
        "n": 8,
        "reason": null,
        "value": 0.2672751738823744
      },
      "landmark": "first_eligible_target_per_patient",
      "n": 8,
      "observations": 8,
      "patients": 8,
      "risk_definition": "1-S(365)",
      "survival_nll": 4.6124675096943974
    },
    "O2-rrt_ensemble/29/H3": {
      "available": true,
      "c_index": {
        "comparable_pairs": 11,
        "reason": null,
        "value": 0.6363636363636364
      },
      "censored": 3,
      "events": 4,
      "horizon": 3,
      "ipcw_brier": {
        "n": 7,
        "reason": null,
        "value": 0.27612153479133644
      },
      "landmark": "first_eligible_target_per_patient",
      "n": 7,
      "observations": 7,
      "patients": 7,
      "risk_definition": "1-S(365)",
      "survival_nll": 4.112970851893936
    },
    "O2-rrt_ensemble/7/H1": {
      "available": true,
      "c_index": {
        "comparable_pairs": 35,
        "reason": null,
        "value": 0.4857142857142857
      },
      "censored": 9,
      "events": 5,
      "horizon": 1,
      "ipcw_brier": {
        "n": 14,
        "reason": null,
        "value": 0.23821470130035563
      },
      "landmark": "first_eligible_target_per_patient",
      "n": 14,
      "observations": 14,
      "patients": 14,
      "risk_definition": "1-S(365)",
      "survival_nll": 3.0378885780061995
    },
    "O2-rrt_ensemble/7/H2": {
      "available": true,
      "c_index": {
        "comparable_pairs": 15,
        "reason": null,
        "value": 0.4
      },
      "censored": 3,
      "events": 5,
      "horizon": 2,
      "ipcw_brier": {
        "n": 8,
        "reason": null,
        "value": 0.26523601680858266
      },
      "landmark": "first_eligible_target_per_patient",
      "n": 8,
      "observations": 8,
      "patients": 8,
      "risk_definition": "1-S(365)",
      "survival_nll": 4.676223098300397
    },
    "O2-rrt_ensemble/7/H3": {
      "available": true,
      "c_index": {
        "comparable_pairs": 11,
        "reason": null,
        "value": 0.45454545454545453
      },
      "censored": 3,
      "events": 4,
      "horizon": 3,
      "ipcw_brier": {
        "n": 7,
        "reason": null,
        "value": 0.2768379753911928
      },
      "landmark": "first_eligible_target_per_patient",
      "n": 7,
      "observations": 7,
      "patients": 7,
      "risk_definition": "1-S(365)",
      "survival_nll": 4.130305751093796
    }
  },
  "validation": {
    "O0/17": {
      "all_landmarks_supplemental": {
        "available": true,
        "c_index": {
          "comparable_pairs": 917,
          "reason": null,
          "value": 0.727917121046892
        },
        "censored": 38,
        "events": 29,
        "ipcw_brier": {
          "n": 67,
          "reason": null,
          "value": 0.16825014978771954
        },
        "landmark": "all_eligible_supplemental",
        "n": 67,
        "observations": 67,
        "patients": 22,
        "repeated_observations": 45,
        "risk_definition": "1-S(365)",
        "survival_nll": 2.8966515201296823
      },
      "primary": {
        "available": true,
        "c_index": {
          "comparable_pairs": 74,
          "reason": null,
          "value": 0.5945945945945946
        },
        "censored": 14,
        "events": 8,
        "ipcw_brier": {
          "n": 22,
          "reason": null,
          "value": 0.21066305453765446
        },
        "landmark": "first_eligible",
        "n": 22,
        "observations": 22,
        "patients": 22,
        "repeated_observations": 0,
        "risk_definition": "1-S(365)",
        "survival_nll": 2.6753143801946533
      }
    },
    "O0/29": {
      "all_landmarks_supplemental": {
        "available": true,
        "c_index": {
          "comparable_pairs": 917,
          "reason": null,
          "value": 0.757360959651036
        },
        "censored": 38,
        "events": 29,
        "ipcw_brier": {
          "n": 67,
          "reason": null,
          "value": 0.15725456940740773
        },
        "landmark": "all_eligible_supplemental",
        "n": 67,
        "observations": 67,
        "patients": 22,
        "repeated_observations": 45,
        "risk_definition": "1-S(365)",
        "survival_nll": 2.9086848115687496
      },
      "primary": {
        "available": true,
        "c_index": {
          "comparable_pairs": 74,
          "reason": null,
          "value": 0.6621621621621622
        },
        "censored": 14,
        "events": 8,
        "ipcw_brier": {
          "n": 22,
          "reason": null,
          "value": 0.21046469888540986
        },
        "landmark": "first_eligible",
        "n": 22,
        "observations": 22,
        "patients": 22,
        "repeated_observations": 0,
        "risk_definition": "1-S(365)",
        "survival_nll": 2.6813142460093577
      }
    },
    "O0/7": {
      "all_landmarks_supplemental": {
        "available": true,
        "c_index": {
          "comparable_pairs": 917,
          "reason": null,
          "value": 0.7453653217011995
        },
        "censored": 38,
        "events": 29,
        "ipcw_brier": {
          "n": 67,
          "reason": null,
          "value": 0.1577518251123071
        },
        "landmark": "all_eligible_supplemental",
        "n": 67,
        "observations": 67,
        "patients": 22,
        "repeated_observations": 45,
        "risk_definition": "1-S(365)",
        "survival_nll": 2.8903276771793505
      },
      "primary": {
        "available": true,
        "c_index": {
          "comparable_pairs": 74,
          "reason": null,
          "value": 0.6216216216216216
        },
        "censored": 14,
        "events": 8,
        "ipcw_brier": {
          "n": 22,
          "reason": null,
          "value": 0.21207246011957206
        },
        "landmark": "first_eligible",
        "n": 22,
        "observations": 22,
        "patients": 22,
        "repeated_observations": 0,
        "risk_definition": "1-S(365)",
        "survival_nll": 2.6836065803442826
      }
    },
    "O1/17": {
      "all_landmarks_supplemental": {
        "available": true,
        "c_index": {
          "comparable_pairs": 917,
          "reason": null,
          "value": 0.5681570338058888
        },
        "censored": 38,
        "events": 29,
        "ipcw_brier": {
          "n": 67,
          "reason": null,
          "value": 0.1818447477058228
        },
        "landmark": "all_eligible_supplemental",
        "n": 67,
        "observations": 67,
        "patients": 22,
        "repeated_observations": 45,
        "risk_definition": "1-S(365)",
        "survival_nll": 3.0692641494879083
      },
      "primary": {
        "available": true,
        "c_index": {
          "comparable_pairs": 74,
          "reason": null,
          "value": 0.3783783783783784
        },
        "censored": 14,
        "events": 8,
        "ipcw_brier": {
          "n": 22,
          "reason": null,
          "value": 0.2658252196248807
        },
        "landmark": "first_eligible",
        "n": 22,
        "observations": 22,
        "patients": 22,
        "repeated_observations": 0,
        "risk_definition": "1-S(365)",
        "survival_nll": 2.7863617171956734
      }
    },
    "O1/29": {
      "all_landmarks_supplemental": {
        "available": true,
        "c_index": {
          "comparable_pairs": 917,
          "reason": null,
          "value": 0.5888767720828789
        },
        "censored": 38,
        "events": 29,
        "ipcw_brier": {
          "n": 67,
          "reason": null,
          "value": 0.18791937970948278
        },
        "landmark": "all_eligible_supplemental",
        "n": 67,
        "observations": 67,
        "patients": 22,
        "repeated_observations": 45,
        "risk_definition": "1-S(365)",
        "survival_nll": 3.0770484359462316
      },
      "primary": {
        "available": true,
        "c_index": {
          "comparable_pairs": 74,
          "reason": null,
          "value": 0.40540540540540543
        },
        "censored": 14,
        "events": 8,
        "ipcw_brier": {
          "n": 22,
          "reason": null,
          "value": 0.26896369225769806
        },
        "landmark": "first_eligible",
        "n": 22,
        "observations": 22,
        "patients": 22,
        "repeated_observations": 0,
        "risk_definition": "1-S(365)",
        "survival_nll": 2.777785973623395
      }
    },
    "O1/7": {
      "all_landmarks_supplemental": {
        "available": true,
        "c_index": {
          "comparable_pairs": 917,
          "reason": null,
          "value": 0.608505997818975
        },
        "censored": 38,
        "events": 29,
        "ipcw_brier": {
          "n": 67,
          "reason": null,
          "value": 0.18186954559794005
        },
        "landmark": "all_eligible_supplemental",
        "n": 67,
        "observations": 67,
        "patients": 22,
        "repeated_observations": 45,
        "risk_definition": "1-S(365)",
        "survival_nll": 3.0503708289991787
      },
      "primary": {
        "available": true,
        "c_index": {
          "comparable_pairs": 74,
          "reason": null,
          "value": 0.4594594594594595
        },
        "censored": 14,
        "events": 8,
        "ipcw_brier": {
          "n": 22,
          "reason": null,
          "value": 0.2516003802431801
        },
        "landmark": "first_eligible",
        "n": 22,
        "observations": 22,
        "patients": 22,
        "repeated_observations": 0,
        "risk_definition": "1-S(365)",
        "survival_nll": 2.7297141934660347
      }
    },
    "O2-baseline/17/H1": {
      "available": true,
      "c_index": {
        "comparable_pairs": 69,
        "reason": null,
        "value": 0.42028985507246375
      },
      "censored": 11,
      "events": 8,
      "horizon": 1,
      "ipcw_brier": {
        "n": 19,
        "reason": null,
        "value": 0.18912663664458748
      },
      "landmark": "first_eligible_target_per_patient",
      "n": 19,
      "observations": 19,
      "patients": 19,
      "risk_definition": "1-S(365)",
      "survival_nll": 2.9650755748152733
    },
    "O2-baseline/17/H2": {
      "available": true,
      "c_index": {
        "comparable_pairs": 24,
        "reason": null,
        "value": 0.6666666666666666
      },
      "censored": 5,
      "events": 5,
      "horizon": 2,
      "ipcw_brier": {
        "n": 10,
        "reason": null,
        "value": 0.17248557050037525
      },
      "landmark": "first_eligible_target_per_patient",
      "n": 10,
      "observations": 10,
      "patients": 10,
      "risk_definition": "1-S(365)",
      "survival_nll": 3.344463162124157
    },
    "O2-baseline/17/H3": {
      "available": true,
      "c_index": {
        "comparable_pairs": 9,
        "reason": null,
        "value": 0.7777777777777778
      },
      "censored": 5,
      "events": 3,
      "horizon": 3,
      "ipcw_brier": {
        "n": 8,
        "reason": null,
        "value": 0.10519846600948807
      },
      "landmark": "first_eligible_target_per_patient",
      "n": 8,
      "observations": 8,
      "patients": 8,
      "risk_definition": "1-S(365)",
      "survival_nll": 2.5001102150417864
    },
    "O2-baseline/29/H1": {
      "available": true,
      "c_index": {
        "comparable_pairs": 69,
        "reason": null,
        "value": 0.37681159420289856
      },
      "censored": 11,
      "events": 8,
      "horizon": 1,
      "ipcw_brier": {
        "n": 19,
        "reason": null,
        "value": 0.20586477265247086
      },
      "landmark": "first_eligible_target_per_patient",
      "n": 19,
      "observations": 19,
      "patients": 19,
      "risk_definition": "1-S(365)",
      "survival_nll": 3.014576169172008
    },
    "O2-baseline/29/H2": {
      "available": true,
      "c_index": {
        "comparable_pairs": 24,
        "reason": null,
        "value": 0.6666666666666666
      },
      "censored": 5,
      "events": 5,
      "horizon": 2,
      "ipcw_brier": {
        "n": 10,
        "reason": null,
        "value": 0.17382390823247734
      },
      "landmark": "first_eligible_target_per_patient",
      "n": 10,
      "observations": 10,
      "patients": 10,
      "risk_definition": "1-S(365)",
      "survival_nll": 3.3225566908717155
    },
    "O2-baseline/29/H3": {
      "available": true,
      "c_index": {
        "comparable_pairs": 9,
        "reason": null,
        "value": 0.8888888888888888
      },
      "censored": 5,
      "events": 3,
      "horizon": 3,
      "ipcw_brier": {
        "n": 8,
        "reason": null,
        "value": 0.0995454401673315
      },
      "landmark": "first_eligible_target_per_patient",
      "n": 8,
      "observations": 8,
      "patients": 8,
      "risk_definition": "1-S(365)",
      "survival_nll": 2.585247617214918
    },
    "O2-baseline/7/H1": {
      "available": true,
      "c_index": {
        "comparable_pairs": 69,
        "reason": null,
        "value": 0.4057971014492754
      },
      "censored": 11,
      "events": 8,
      "horizon": 1,
      "ipcw_brier": {
        "n": 19,
        "reason": null,
        "value": 0.19416324554824763
      },
      "landmark": "first_eligible_target_per_patient",
      "n": 19,
      "observations": 19,
      "patients": 19,
      "risk_definition": "1-S(365)",
      "survival_nll": 2.9926823491328642
    },
    "O2-baseline/7/H2": {
      "available": true,
      "c_index": {
        "comparable_pairs": 24,
        "reason": null,
        "value": 0.75
      },
      "censored": 5,
      "events": 5,
      "horizon": 2,
      "ipcw_brier": {
        "n": 10,
        "reason": null,
        "value": 0.16139643402596643
      },
      "landmark": "first_eligible_target_per_patient",
      "n": 10,
      "observations": 10,
      "patients": 10,
      "risk_definition": "1-S(365)",
      "survival_nll": 3.305292657017708
    },
    "O2-baseline/7/H3": {
      "available": true,
      "c_index": {
        "comparable_pairs": 9,
        "reason": null,
        "value": 0.8888888888888888
      },
      "censored": 5,
      "events": 3,
      "horizon": 3,
      "ipcw_brier": {
        "n": 8,
        "reason": null,
        "value": 0.11031826709008974
      },
      "landmark": "first_eligible_target_per_patient",
      "n": 8,
      "observations": 8,
      "patients": 8,
      "risk_definition": "1-S(365)",
      "survival_nll": 2.587088323198259
    },
    "O2-ensemble/17/H1": {
      "available": true,
      "c_index": {
        "comparable_pairs": 69,
        "reason": null,
        "value": 0.4057971014492754
      },
      "censored": 11,
      "events": 8,
      "horizon": 1,
      "ipcw_brier": {
        "n": 19,
        "reason": null,
        "value": 0.18543147338228072
      },
      "landmark": "first_eligible_target_per_patient",
      "n": 19,
      "observations": 19,
      "patients": 19,
      "risk_definition": "1-S(365)",
      "survival_nll": 2.972821085468719
    },
    "O2-ensemble/17/H2": {
      "available": true,
      "c_index": {
        "comparable_pairs": 24,
        "reason": null,
        "value": 0.625
      },
      "censored": 5,
      "events": 5,
      "horizon": 2,
      "ipcw_brier": {
        "n": 10,
        "reason": null,
        "value": 0.16773048678954336
      },
      "landmark": "first_eligible_target_per_patient",
      "n": 10,
      "observations": 10,
      "patients": 10,
      "risk_definition": "1-S(365)",
      "survival_nll": 3.360100420564413
    },
    "O2-ensemble/17/H3": {
      "available": true,
      "c_index": {
        "comparable_pairs": 9,
        "reason": null,
        "value": 0.8888888888888888
      },
      "censored": 5,
      "events": 3,
      "horizon": 3,
      "ipcw_brier": {
        "n": 8,
        "reason": null,
        "value": 0.10566740413335969
      },
      "landmark": "first_eligible_target_per_patient",
      "n": 8,
      "observations": 8,
      "patients": 8,
      "risk_definition": "1-S(365)",
      "survival_nll": 2.5125194150023162
    },
    "O2-ensemble/29/H1": {
      "available": true,
      "c_index": {
        "comparable_pairs": 69,
        "reason": null,
        "value": 0.391304347826087
      },
      "censored": 11,
      "events": 8,
      "horizon": 1,
      "ipcw_brier": {
        "n": 19,
        "reason": null,
        "value": 0.20123587727266481
      },
      "landmark": "first_eligible_target_per_patient",
      "n": 19,
      "observations": 19,
      "patients": 19,
      "risk_definition": "1-S(365)",
      "survival_nll": 3.0344785612664724
    },
    "O2-ensemble/29/H2": {
      "available": true,
      "c_index": {
        "comparable_pairs": 24,
        "reason": null,
        "value": 0.625
      },
      "censored": 5,
      "events": 5,
      "horizon": 2,
      "ipcw_brier": {
        "n": 10,
        "reason": null,
        "value": 0.16732170870586438
      },
      "landmark": "first_eligible_target_per_patient",
      "n": 10,
      "observations": 10,
      "patients": 10,
      "risk_definition": "1-S(365)",
      "survival_nll": 3.39351766705513
    },
    "O2-ensemble/29/H3": {
      "available": true,
      "c_index": {
        "comparable_pairs": 9,
        "reason": null,
        "value": 0.7777777777777778
      },
      "censored": 5,
      "events": 3,
      "horizon": 3,
      "ipcw_brier": {
        "n": 8,
        "reason": null,
        "value": 0.105941571893643
      },
      "landmark": "first_eligible_target_per_patient",
      "n": 8,
      "observations": 8,
      "patients": 8,
      "risk_definition": "1-S(365)",
      "survival_nll": 2.591723396908492
    },
    "O2-ensemble/7/H1": {
      "available": true,
      "c_index": {
        "comparable_pairs": 69,
        "reason": null,
        "value": 0.391304347826087
      },
      "censored": 11,
      "events": 8,
      "horizon": 1,
      "ipcw_brier": {
        "n": 19,
        "reason": null,
        "value": 0.18986222526707416
      },
      "landmark": "first_eligible_target_per_patient",
      "n": 19,
      "observations": 19,
      "patients": 19,
      "risk_definition": "1-S(365)",
      "survival_nll": 3.018659500454209
    },
    "O2-ensemble/7/H2": {
      "available": true,
      "c_index": {
        "comparable_pairs": 24,
        "reason": null,
        "value": 0.6666666666666666
      },
      "censored": 5,
      "events": 5,
      "horizon": 2,
      "ipcw_brier": {
        "n": 10,
        "reason": null,
        "value": 0.1514275670691555
      },
      "landmark": "first_eligible_target_per_patient",
      "n": 10,
      "observations": 10,
      "patients": 10,
      "risk_definition": "1-S(365)",
      "survival_nll": 3.337055142223835
    },
    "O2-ensemble/7/H3": {
      "available": true,
      "c_index": {
        "comparable_pairs": 9,
        "reason": null,
        "value": 1.0
      },
      "censored": 5,
      "events": 3,
      "horizon": 3,
      "ipcw_brier": {
        "n": 8,
        "reason": null,
        "value": 0.11625470043103771
      },
      "landmark": "first_eligible_target_per_patient",
      "n": 8,
      "observations": 8,
      "patients": 8,
      "risk_definition": "1-S(365)",
      "survival_nll": 2.577197966631502
    },
    "O2-rrt/17/H1": {
      "available": true,
      "c_index": {
        "comparable_pairs": 69,
        "reason": null,
        "value": 0.42028985507246375
      },
      "censored": 11,
      "events": 8,
      "horizon": 1,
      "ipcw_brier": {
        "n": 19,
        "reason": null,
        "value": 0.19728652353575168
      },
      "landmark": "first_eligible_target_per_patient",
      "n": 19,
      "observations": 19,
      "patients": 19,
      "risk_definition": "1-S(365)",
      "survival_nll": 2.9575639885703198
    },
    "O2-rrt/17/H2": {
      "available": true,
      "c_index": {
        "comparable_pairs": 24,
        "reason": null,
        "value": 0.625
      },
      "censored": 5,
      "events": 5,
      "horizon": 2,
      "ipcw_brier": {
        "n": 10,
        "reason": null,
        "value": 0.1805833211812023
      },
      "landmark": "first_eligible_target_per_patient",
      "n": 10,
      "observations": 10,
      "patients": 10,
      "risk_definition": "1-S(365)",
      "survival_nll": 3.309587623924017
    },
    "O2-rrt/17/H3": {
      "available": true,
      "c_index": {
        "comparable_pairs": 9,
        "reason": null,
        "value": 0.7777777777777778
      },
      "censored": 5,
      "events": 3,
      "horizon": 3,
      "ipcw_brier": {
        "n": 8,
        "reason": null,
        "value": 0.10212260368730028
      },
      "landmark": "first_eligible_target_per_patient",
      "n": 8,
      "observations": 8,
      "patients": 8,
      "risk_definition": "1-S(365)",
      "survival_nll": 2.5344717586413026
    },
    "O2-rrt/29/H1": {
      "available": true,
      "c_index": {
        "comparable_pairs": 69,
        "reason": null,
        "value": 0.4057971014492754
      },
      "censored": 11,
      "events": 8,
      "horizon": 1,
      "ipcw_brier": {
        "n": 19,
        "reason": null,
        "value": 0.21021785132387472
      },
      "landmark": "first_eligible_target_per_patient",
      "n": 19,
      "observations": 19,
      "patients": 19,
      "risk_definition": "1-S(365)",
      "survival_nll": 3.0087805881017915
    },
    "O2-rrt/29/H2": {
      "available": true,
      "c_index": {
        "comparable_pairs": 24,
        "reason": null,
        "value": 0.7083333333333334
      },
      "censored": 5,
      "events": 5,
      "horizon": 2,
      "ipcw_brier": {
        "n": 10,
        "reason": null,
        "value": 0.1755023778434905
      },
      "landmark": "first_eligible_target_per_patient",
      "n": 10,
      "observations": 10,
      "patients": 10,
      "risk_definition": "1-S(365)",
      "survival_nll": 3.3283914744853975
    },
    "O2-rrt/29/H3": {
      "available": true,
      "c_index": {
        "comparable_pairs": 9,
        "reason": null,
        "value": 0.8888888888888888
      },
      "censored": 5,
      "events": 3,
      "horizon": 3,
      "ipcw_brier": {
        "n": 8,
        "reason": null,
        "value": 0.09926676479565265
      },
      "landmark": "first_eligible_target_per_patient",
      "n": 8,
      "observations": 8,
      "patients": 8,
      "risk_definition": "1-S(365)",
      "survival_nll": 2.582573269493878
    },
    "O2-rrt/7/H1": {
      "available": true,
      "c_index": {
        "comparable_pairs": 69,
        "reason": null,
        "value": 0.37681159420289856
      },
      "censored": 11,
      "events": 8,
      "horizon": 1,
      "ipcw_brier": {
        "n": 19,
        "reason": null,
        "value": 0.19969959055717498
      },
      "landmark": "first_eligible_target_per_patient",
      "n": 19,
      "observations": 19,
      "patients": 19,
      "risk_definition": "1-S(365)",
      "survival_nll": 2.9987941747531295
    },
    "O2-rrt/7/H2": {
      "available": true,
      "c_index": {
        "comparable_pairs": 24,
        "reason": null,
        "value": 0.7083333333333334
      },
      "censored": 5,
      "events": 5,
      "horizon": 2,
      "ipcw_brier": {
        "n": 10,
        "reason": null,
        "value": 0.16716253286150312
      },
      "landmark": "first_eligible_target_per_patient",
      "n": 10,
      "observations": 10,
      "patients": 10,
      "risk_definition": "1-S(365)",
      "survival_nll": 3.2975788429379462
    },
    "O2-rrt/7/H3": {
      "available": true,
      "c_index": {
        "comparable_pairs": 9,
        "reason": null,
        "value": 0.8888888888888888
      },
      "censored": 5,
      "events": 3,
      "horizon": 3,
      "ipcw_brier": {
        "n": 8,
        "reason": null,
        "value": 0.10558626593950876
      },
      "landmark": "first_eligible_target_per_patient",
      "n": 8,
      "observations": 8,
      "patients": 8,
      "risk_definition": "1-S(365)",
      "survival_nll": 2.5952527285553515
    },
    "O2-rrt_ensemble/17/H1": {
      "available": true,
      "c_index": {
        "comparable_pairs": 69,
        "reason": null,
        "value": 0.4057971014492754
      },
      "censored": 11,
      "events": 8,
      "horizon": 1,
      "ipcw_brier": {
        "n": 19,
        "reason": null,
        "value": 0.19336988160288607
      },
      "landmark": "first_eligible_target_per_patient",
      "n": 19,
      "observations": 19,
      "patients": 19,
      "risk_definition": "1-S(365)",
      "survival_nll": 2.9591914639367083
    },
    "O2-rrt_ensemble/17/H2": {
      "available": true,
      "c_index": {
        "comparable_pairs": 24,
        "reason": null,
        "value": 0.6666666666666666
      },
      "censored": 5,
      "events": 5,
      "horizon": 2,
      "ipcw_brier": {
        "n": 10,
        "reason": null,
        "value": 0.17421781892666635
      },
      "landmark": "first_eligible_target_per_patient",
      "n": 10,
      "observations": 10,
      "patients": 10,
      "risk_definition": "1-S(365)",
      "survival_nll": 3.3047434210777284
    },
    "O2-rrt_ensemble/17/H3": {
      "available": true,
      "c_index": {
        "comparable_pairs": 9,
        "reason": null,
        "value": 0.8888888888888888
      },
      "censored": 5,
      "events": 3,
      "horizon": 3,
      "ipcw_brier": {
        "n": 8,
        "reason": null,
        "value": 0.09437156323985474
      },
      "landmark": "first_eligible_target_per_patient",
      "n": 8,
      "observations": 8,
      "patients": 8,
      "risk_definition": "1-S(365)",
      "survival_nll": 2.525866073090583
    },
    "O2-rrt_ensemble/29/H1": {
      "available": true,
      "c_index": {
        "comparable_pairs": 69,
        "reason": null,
        "value": 0.391304347826087
      },
      "censored": 11,
      "events": 8,
      "horizon": 1,
      "ipcw_brier": {
        "n": 19,
        "reason": null,
        "value": 0.20559682062930404
      },
      "landmark": "first_eligible_target_per_patient",
      "n": 19,
      "observations": 19,
      "patients": 19,
      "risk_definition": "1-S(365)",
      "survival_nll": 3.012566877087872
    },
    "O2-rrt_ensemble/29/H2": {
      "available": true,
      "c_index": {
        "comparable_pairs": 24,
        "reason": null,
        "value": 0.7083333333333334
      },
      "censored": 5,
      "events": 5,
      "horizon": 2,
      "ipcw_brier": {
        "n": 10,
        "reason": null,
        "value": 0.16256786454885802
      },
      "landmark": "first_eligible_target_per_patient",
      "n": 10,
      "observations": 10,
      "patients": 10,
      "risk_definition": "1-S(365)",
      "survival_nll": 3.3205524161458015
    },
    "O2-rrt_ensemble/29/H3": {
      "available": true,
      "c_index": {
        "comparable_pairs": 9,
        "reason": null,
        "value": 0.8888888888888888
      },
      "censored": 5,
      "events": 3,
      "horizon": 3,
      "ipcw_brier": {
        "n": 8,
        "reason": null,
        "value": 0.09304531023300214
      },
      "landmark": "first_eligible_target_per_patient",
      "n": 8,
      "observations": 8,
      "patients": 8,
      "risk_definition": "1-S(365)",
      "survival_nll": 2.5550031289458275
    },
    "O2-rrt_ensemble/7/H1": {
      "available": true,
      "c_index": {
        "comparable_pairs": 69,
        "reason": null,
        "value": 0.391304347826087
      },
      "censored": 11,
      "events": 8,
      "horizon": 1,
      "ipcw_brier": {
        "n": 19,
        "reason": null,
        "value": 0.20230101645864582
      },
      "landmark": "first_eligible_target_per_patient",
      "n": 19,
      "observations": 19,
      "patients": 19,
      "risk_definition": "1-S(365)",
      "survival_nll": 3.0039318236277293
    },
    "O2-rrt_ensemble/7/H2": {
      "available": true,
      "c_index": {
        "comparable_pairs": 24,
        "reason": null,
        "value": 0.7083333333333334
      },
      "censored": 5,
      "events": 5,
      "horizon": 2,
      "ipcw_brier": {
        "n": 10,
        "reason": null,
        "value": 0.16456499090883006
      },
      "landmark": "first_eligible_target_per_patient",
      "n": 10,
      "observations": 10,
      "patients": 10,
      "risk_definition": "1-S(365)",
      "survival_nll": 3.3052613005042075
    },
    "O2-rrt_ensemble/7/H3": {
      "available": true,
      "c_index": {
        "comparable_pairs": 9,
        "reason": null,
        "value": 0.8888888888888888
      },
      "censored": 5,
      "events": 3,
      "horizon": 3,
      "ipcw_brier": {
        "n": 8,
        "reason": null,
        "value": 0.10782311966745228
      },
      "landmark": "first_eligible_target_per_patient",
      "n": 8,
      "observations": 8,
      "patients": 8,
      "risk_definition": "1-S(365)",
      "survival_nll": 2.5597738097421825
    }
  }
}
```

## Observed replay — treatment agreement only

```json
{
  "test": {
    "fixed_plan/17": {
      "abstain_rate": 0.0,
      "beam_nodes": 3402,
      "both_empty_fraction": 0.3333333333333333,
      "candidate_recall": 0.9365079365079365,
      "catalog_coverage": 0.9841269841269841,
      "conditional_on_recommendation": {
        "f1": 0.4383975812547241,
        "jaccard": 0.4185185185185185,
        "precision": 0.42063492063492064,
        "recall": 0.47354497354497355
      },
      "interpretation": "agreement with recorded treatment; not causal treatment quality",
      "n": 63,
      "plan_revision_evaluable_count": 0,
      "plan_revision_evaluable_fraction": 0.0,
      "plan_revision_interpretation": "behavioral response to new observations; a higher rate is not inherently better",
      "plan_revision_rate": null,
      "plan_revision_reason": "fixed_plan_does_not_replan",
      "policy_calls": 0,
      "recommendation_coverage": 1.0,
      "recommendations": 63,
      "sequential_action_change_count": 41,
      "sequential_action_change_rate": 0.43902439024390244,
      "structural_rule_violation_rate": null,
      "structural_rule_violation_reason": "clinical_rules_disabled",
      "unconditional": {
        "f1": 0.4383975812547241,
        "jaccard": 0.4185185185185185,
        "precision": 0.42063492063492064,
        "recall": 0.47354497354497355
      },
      "wall_time_ms_mean": 1912.5489590380946,
      "world_model_forwards": 17010
    },
    "fixed_plan/29": {
      "abstain_rate": 0.0,
      "beam_nodes": 3402,
      "both_empty_fraction": 0.25396825396825395,
      "candidate_recall": 0.9365079365079365,
      "catalog_coverage": 0.9841269841269841,
      "conditional_on_recommendation": {
        "f1": 0.3833711262282691,
        "jaccard": 0.3611111111111111,
        "precision": 0.376984126984127,
        "recall": 0.40211640211640215
      },
      "interpretation": "agreement with recorded treatment; not causal treatment quality",
      "n": 63,
      "plan_revision_evaluable_count": 0,
      "plan_revision_evaluable_fraction": 0.0,
      "plan_revision_interpretation": "behavioral response to new observations; a higher rate is not inherently better",
      "plan_revision_rate": null,
      "plan_revision_reason": "fixed_plan_does_not_replan",
      "policy_calls": 0,
      "recommendation_coverage": 1.0,
      "recommendations": 63,
      "sequential_action_change_count": 41,
      "sequential_action_change_rate": 0.34146341463414637,
      "structural_rule_violation_rate": null,
      "structural_rule_violation_reason": "clinical_rules_disabled",
      "unconditional": {
        "f1": 0.3833711262282691,
        "jaccard": 0.3611111111111111,
        "precision": 0.376984126984127,
        "recall": 0.40211640211640215
      },
      "wall_time_ms_mean": 1528.1399741449156,
      "world_model_forwards": 17010
    },
    "fixed_plan/7": {
      "abstain_rate": 0.0,
      "beam_nodes": 3402,
      "both_empty_fraction": 0.2222222222222222,
      "candidate_recall": 0.9365079365079365,
      "catalog_coverage": 0.9841269841269841,
      "conditional_on_recommendation": {
        "f1": 0.33053665910808766,
        "jaccard": 0.30687830687830686,
        "precision": 0.32275132275132273,
        "recall": 0.34788359788359785
      },
      "interpretation": "agreement with recorded treatment; not causal treatment quality",
      "n": 63,
      "plan_revision_evaluable_count": 0,
      "plan_revision_evaluable_fraction": 0.0,
      "plan_revision_interpretation": "behavioral response to new observations; a higher rate is not inherently better",
      "plan_revision_rate": null,
      "plan_revision_reason": "fixed_plan_does_not_replan",
      "policy_calls": 0,
      "recommendation_coverage": 1.0,
      "recommendations": 63,
      "sequential_action_change_count": 41,
      "sequential_action_change_rate": 0.1951219512195122,
      "structural_rule_violation_rate": null,
      "structural_rule_violation_reason": "clinical_rules_disabled",
      "unconditional": {
        "f1": 0.33053665910808766,
        "jaccard": 0.30687830687830686,
        "precision": 0.32275132275132273,
        "recall": 0.34788359788359785
      },
      "wall_time_ms_mean": 1973.5707445107105,
      "world_model_forwards": 17010
    },
    "frequency/17": {
      "abstain_rate": 0.0,
      "beam_nodes": 0,
      "both_empty_fraction": 0.5873015873015873,
      "candidate_recall": 0.9365079365079365,
      "catalog_coverage": 0.9841269841269841,
      "conditional_on_recommendation": {
        "f1": 0.5873015873015873,
        "jaccard": 0.5873015873015873,
        "precision": 0.5873015873015873,
        "recall": 0.5873015873015873
      },
      "interpretation": "agreement with recorded treatment; not causal treatment quality",
      "n": 63,
      "plan_revision_evaluable_count": 0,
      "plan_revision_evaluable_fraction": 0.0,
      "plan_revision_interpretation": "behavioral response to new observations; a higher rate is not inherently better",
      "plan_revision_rate": null,
      "plan_revision_reason": "no_prior_future_action_in_h1_policy",
      "policy_calls": 0,
      "recommendation_coverage": 1.0,
      "recommendations": 63,
      "sequential_action_change_count": 41,
      "sequential_action_change_rate": 0.0,
      "structural_rule_violation_rate": null,
      "structural_rule_violation_reason": "clinical_rules_disabled",
      "unconditional": {
        "f1": 0.5873015873015873,
        "jaccard": 0.5873015873015873,
        "precision": 0.5873015873015873,
        "recall": 0.5873015873015873
      },
      "wall_time_ms_mean": 0.18056343117403606,
      "world_model_forwards": 0
    },
    "frequency/29": {
      "abstain_rate": 0.0,
      "beam_nodes": 0,
      "both_empty_fraction": 0.5873015873015873,
      "candidate_recall": 0.9365079365079365,
      "catalog_coverage": 0.9841269841269841,
      "conditional_on_recommendation": {
        "f1": 0.5873015873015873,
        "jaccard": 0.5873015873015873,
        "precision": 0.5873015873015873,
        "recall": 0.5873015873015873
      },
      "interpretation": "agreement with recorded treatment; not causal treatment quality",
      "n": 63,
      "plan_revision_evaluable_count": 0,
      "plan_revision_evaluable_fraction": 0.0,
      "plan_revision_interpretation": "behavioral response to new observations; a higher rate is not inherently better",
      "plan_revision_rate": null,
      "plan_revision_reason": "no_prior_future_action_in_h1_policy",
      "policy_calls": 0,
      "recommendation_coverage": 1.0,
      "recommendations": 63,
      "sequential_action_change_count": 41,
      "sequential_action_change_rate": 0.0,
      "structural_rule_violation_rate": null,
      "structural_rule_violation_reason": "clinical_rules_disabled",
      "unconditional": {
        "f1": 0.5873015873015873,
        "jaccard": 0.5873015873015873,
        "precision": 0.5873015873015873,
        "recall": 0.5873015873015873
      },
      "wall_time_ms_mean": 0.1816240643771986,
      "world_model_forwards": 0
    },
    "frequency/7": {
      "abstain_rate": 0.0,
      "beam_nodes": 0,
      "both_empty_fraction": 0.5873015873015873,
      "candidate_recall": 0.9365079365079365,
      "catalog_coverage": 0.9841269841269841,
      "conditional_on_recommendation": {
        "f1": 0.5873015873015873,
        "jaccard": 0.5873015873015873,
        "precision": 0.5873015873015873,
        "recall": 0.5873015873015873
      },
      "interpretation": "agreement with recorded treatment; not causal treatment quality",
      "n": 63,
      "plan_revision_evaluable_count": 0,
      "plan_revision_evaluable_fraction": 0.0,
      "plan_revision_interpretation": "behavioral response to new observations; a higher rate is not inherently better",
      "plan_revision_rate": null,
      "plan_revision_reason": "no_prior_future_action_in_h1_policy",
      "policy_calls": 0,
      "recommendation_coverage": 1.0,
      "recommendations": 63,
      "sequential_action_change_count": 41,
      "sequential_action_change_rate": 0.0,
      "structural_rule_violation_rate": null,
      "structural_rule_violation_reason": "clinical_rules_disabled",
      "unconditional": {
        "f1": 0.5873015873015873,
        "jaccard": 0.5873015873015873,
        "precision": 0.5873015873015873,
        "recall": 0.5873015873015873
      },
      "wall_time_ms_mean": 0.25707437834214597,
      "world_model_forwards": 0
    },
    "greedy/17": {
      "abstain_rate": 0.0,
      "beam_nodes": 378,
      "both_empty_fraction": 0.047619047619047616,
      "candidate_recall": 0.9365079365079365,
      "catalog_coverage": 0.9841269841269841,
      "conditional_on_recommendation": {
        "f1": 0.20642479213907783,
        "jaccard": 0.1724867724867725,
        "precision": 0.18386243386243384,
        "recall": 0.2513227513227513
      },
      "interpretation": "agreement with recorded treatment; not causal treatment quality",
      "n": 63,
      "plan_revision_evaluable_count": 0,
      "plan_revision_evaluable_fraction": 0.0,
      "plan_revision_interpretation": "behavioral response to new observations; a higher rate is not inherently better",
      "plan_revision_rate": null,
      "plan_revision_reason": "no_prior_future_action_in_h1_policy",
      "policy_calls": 0,
      "recommendation_coverage": 1.0,
      "recommendations": 63,
      "sequential_action_change_count": 41,
      "sequential_action_change_rate": 0.4146341463414634,
      "structural_rule_violation_rate": null,
      "structural_rule_violation_reason": "clinical_rules_disabled",
      "unconditional": {
        "f1": 0.20642479213907783,
        "jaccard": 0.1724867724867725,
        "precision": 0.18386243386243384,
        "recall": 0.2513227513227513
      },
      "wall_time_ms_mean": 221.41288027134058,
      "world_model_forwards": 1890
    },
    "greedy/29": {
      "abstain_rate": 0.0,
      "beam_nodes": 378,
      "both_empty_fraction": 0.25396825396825395,
      "candidate_recall": 0.9365079365079365,
      "catalog_coverage": 0.9841269841269841,
      "conditional_on_recommendation": {
        "f1": 0.39644746787603935,
        "jaccard": 0.3693121693121693,
        "precision": 0.38492063492063494,
        "recall": 0.42592592592592593
      },
      "interpretation": "agreement with recorded treatment; not causal treatment quality",
      "n": 63,
      "plan_revision_evaluable_count": 0,
      "plan_revision_evaluable_fraction": 0.0,
      "plan_revision_interpretation": "behavioral response to new observations; a higher rate is not inherently better",
      "plan_revision_rate": null,
      "plan_revision_reason": "no_prior_future_action_in_h1_policy",
      "policy_calls": 0,
      "recommendation_coverage": 1.0,
      "recommendations": 63,
      "sequential_action_change_count": 41,
      "sequential_action_change_rate": 0.12195121951219512,
      "structural_rule_violation_rate": null,
      "structural_rule_violation_reason": "clinical_rules_disabled",
      "unconditional": {
        "f1": 0.39644746787603935,
        "jaccard": 0.3693121693121693,
        "precision": 0.38492063492063494,
        "recall": 0.42592592592592593
      },
      "wall_time_ms_mean": 219.31723044401716,
      "world_model_forwards": 1890
    },
    "greedy/7": {
      "abstain_rate": 0.0,
      "beam_nodes": 378,
      "both_empty_fraction": 0.25396825396825395,
      "candidate_recall": 0.9365079365079365,
      "catalog_coverage": 0.9841269841269841,
      "conditional_on_recommendation": {
        "f1": 0.3826908541194255,
        "jaccard": 0.35714285714285715,
        "precision": 0.37566137566137564,
        "recall": 0.3994708994708994
      },
      "interpretation": "agreement with recorded treatment; not causal treatment quality",
      "n": 63,
      "plan_revision_evaluable_count": 0,
      "plan_revision_evaluable_fraction": 0.0,
      "plan_revision_interpretation": "behavioral response to new observations; a higher rate is not inherently better",
      "plan_revision_rate": null,
      "plan_revision_reason": "no_prior_future_action_in_h1_policy",
      "policy_calls": 0,
      "recommendation_coverage": 1.0,
      "recommendations": 63,
      "sequential_action_change_count": 41,
      "sequential_action_change_rate": 0.2682926829268293,
      "structural_rule_violation_rate": null,
      "structural_rule_violation_reason": "clinical_rules_disabled",
      "unconditional": {
        "f1": 0.3826908541194255,
        "jaccard": 0.35714285714285715,
        "precision": 0.37566137566137564,
        "recall": 0.3994708994708994
      },
      "wall_time_ms_mean": 230.88551579516323,
      "world_model_forwards": 1890
    },
    "mpc_ensemble/17": {
      "abstain_rate": 0.0,
      "beam_nodes": 3402,
      "both_empty_fraction": 0.47619047619047616,
      "candidate_recall": 0.9365079365079365,
      "catalog_coverage": 0.9841269841269841,
      "conditional_on_recommendation": {
        "f1": 0.5029478458049886,
        "jaccard": 0.4939153439153439,
        "precision": 0.4973544973544973,
        "recall": 0.5132275132275131
      },
      "interpretation": "agreement with recorded treatment; not causal treatment quality",
      "n": 63,
      "plan_revision_evaluable_count": 41,
      "plan_revision_evaluable_fraction": 0.6507936507936508,
      "plan_revision_interpretation": "behavioral response to new observations; a higher rate is not inherently better",
      "plan_revision_rate": 0.2926829268292683,
      "plan_revision_reason": null,
      "policy_calls": 0,
      "recommendation_coverage": 1.0,
      "recommendations": 63,
      "sequential_action_change_count": 41,
      "sequential_action_change_rate": 0.12195121951219512,
      "structural_rule_violation_rate": null,
      "structural_rule_violation_reason": "clinical_rules_disabled",
      "unconditional": {
        "f1": 0.5029478458049886,
        "jaccard": 0.4939153439153439,
        "precision": 0.4973544973544973,
        "recall": 0.5132275132275131
      },
      "wall_time_ms_mean": 1915.8565437787079,
      "world_model_forwards": 17010
    },
    "mpc_ensemble/29": {
      "abstain_rate": 0.0,
      "beam_nodes": 3402,
      "both_empty_fraction": 0.3492063492063492,
      "candidate_recall": 0.9365079365079365,
      "catalog_coverage": 0.9841269841269841,
      "conditional_on_recommendation": {
        "f1": 0.40899470899470897,
        "jaccard": 0.3968253968253968,
        "precision": 0.41798941798941797,
        "recall": 0.40476190476190477
      },
      "interpretation": "agreement with recorded treatment; not causal treatment quality",
      "n": 63,
      "plan_revision_evaluable_count": 41,
      "plan_revision_evaluable_fraction": 0.6507936507936508,
      "plan_revision_interpretation": "behavioral response to new observations; a higher rate is not inherently better",
      "plan_revision_rate": 0.4634146341463415,
      "plan_revision_reason": null,
      "policy_calls": 0,
      "recommendation_coverage": 1.0,
      "recommendations": 63,
      "sequential_action_change_count": 41,
      "sequential_action_change_rate": 0.43902439024390244,
      "structural_rule_violation_rate": null,
      "structural_rule_violation_reason": "clinical_rules_disabled",
      "unconditional": {
        "f1": 0.40899470899470897,
        "jaccard": 0.3968253968253968,
        "precision": 0.41798941798941797,
        "recall": 0.40476190476190477
      },
      "wall_time_ms_mean": 1922.3073351174771,
      "world_model_forwards": 17010
    },
    "mpc_ensemble/7": {
      "abstain_rate": 0.0,
      "beam_nodes": 3402,
      "both_empty_fraction": 0.3333333333333333,
      "candidate_recall": 0.9365079365079365,
      "catalog_coverage": 0.9841269841269841,
      "conditional_on_recommendation": {
        "f1": 0.409901738473167,
        "jaccard": 0.39814814814814814,
        "precision": 0.4074074074074074,
        "recall": 0.41931216931216925
      },
      "interpretation": "agreement with recorded treatment; not causal treatment quality",
      "n": 63,
      "plan_revision_evaluable_count": 41,
      "plan_revision_evaluable_fraction": 0.6507936507936508,
      "plan_revision_interpretation": "behavioral response to new observations; a higher rate is not inherently better",
      "plan_revision_rate": 0.2926829268292683,
      "plan_revision_reason": null,
      "policy_calls": 0,
      "recommendation_coverage": 1.0,
      "recommendations": 63,
      "sequential_action_change_count": 41,
      "sequential_action_change_rate": 0.2926829268292683,
      "structural_rule_violation_rate": null,
      "structural_rule_violation_reason": "clinical_rules_disabled",
      "unconditional": {
        "f1": 0.409901738473167,
        "jaccard": 0.39814814814814814,
        "precision": 0.4074074074074074,
        "recall": 0.41931216931216925
      },
      "wall_time_ms_mean": 1984.2387006566341,
      "world_model_forwards": 17010
    },
    "mpc_rrt_ensemble/17": {
      "abstain_rate": 0.0,
      "beam_nodes": 3402,
      "both_empty_fraction": 0.47619047619047616,
      "candidate_recall": 0.9365079365079365,
      "catalog_coverage": 0.9841269841269841,
      "conditional_on_recommendation": {
        "f1": 0.5346938775510204,
        "jaccard": 0.5177248677248677,
        "precision": 0.521164021164021,
        "recall": 0.5608465608465608
      },
      "interpretation": "agreement with recorded treatment; not causal treatment quality",
      "n": 63,
      "plan_revision_evaluable_count": 41,
      "plan_revision_evaluable_fraction": 0.6507936507936508,
      "plan_revision_interpretation": "behavioral response to new observations; a higher rate is not inherently better",
      "plan_revision_rate": 0.2682926829268293,
      "plan_revision_reason": null,
      "policy_calls": 0,
      "recommendation_coverage": 1.0,
      "recommendations": 63,
      "sequential_action_change_count": 41,
      "sequential_action_change_rate": 0.1951219512195122,
      "structural_rule_violation_rate": null,
      "structural_rule_violation_reason": "clinical_rules_disabled",
      "unconditional": {
        "f1": 0.5346938775510204,
        "jaccard": 0.5177248677248677,
        "precision": 0.521164021164021,
        "recall": 0.5608465608465608
      },
      "wall_time_ms_mean": 1914.8237361278873,
      "world_model_forwards": 17010
    },
    "mpc_rrt_ensemble/29": {
      "abstain_rate": 0.0,
      "beam_nodes": 3402,
      "both_empty_fraction": 0.2698412698412698,
      "candidate_recall": 0.9365079365079365,
      "catalog_coverage": 0.9841269841269841,
      "conditional_on_recommendation": {
        "f1": 0.3507936507936508,
        "jaccard": 0.3412698412698413,
        "precision": 0.35714285714285715,
        "recall": 0.3465608465608466
      },
      "interpretation": "agreement with recorded treatment; not causal treatment quality",
      "n": 63,
      "plan_revision_evaluable_count": 41,
      "plan_revision_evaluable_fraction": 0.6507936507936508,
      "plan_revision_interpretation": "behavioral response to new observations; a higher rate is not inherently better",
      "plan_revision_rate": 0.4146341463414634,
      "plan_revision_reason": null,
      "policy_calls": 0,
      "recommendation_coverage": 1.0,
      "recommendations": 63,
      "sequential_action_change_count": 41,
      "sequential_action_change_rate": 0.2926829268292683,
      "structural_rule_violation_rate": null,
      "structural_rule_violation_reason": "clinical_rules_disabled",
      "unconditional": {
        "f1": 0.3507936507936508,
        "jaccard": 0.3412698412698413,
        "precision": 0.35714285714285715,
        "recall": 0.3465608465608466
      },
      "wall_time_ms_mean": 1937.2657790645173,
      "world_model_forwards": 17010
    },
    "mpc_rrt_ensemble/7": {
      "abstain_rate": 0.0,
      "beam_nodes": 3402,
      "both_empty_fraction": 0.2698412698412698,
      "candidate_recall": 0.9365079365079365,
      "catalog_coverage": 0.9841269841269841,
      "conditional_on_recommendation": {
        "f1": 0.35910808767951624,
        "jaccard": 0.34259259259259256,
        "precision": 0.3544973544973545,
        "recall": 0.37169312169312163
      },
      "interpretation": "agreement with recorded treatment; not causal treatment quality",
      "n": 63,
      "plan_revision_evaluable_count": 41,
      "plan_revision_evaluable_fraction": 0.6507936507936508,
      "plan_revision_interpretation": "behavioral response to new observations; a higher rate is not inherently better",
      "plan_revision_rate": 0.36585365853658536,
      "plan_revision_reason": null,
      "policy_calls": 0,
      "recommendation_coverage": 1.0,
      "recommendations": 63,
      "sequential_action_change_count": 41,
      "sequential_action_change_rate": 0.3170731707317073,
      "structural_rule_violation_rate": null,
      "structural_rule_violation_reason": "clinical_rules_disabled",
      "unconditional": {
        "f1": 0.35910808767951624,
        "jaccard": 0.34259259259259256,
        "precision": 0.3544973544973545,
        "recall": 0.37169312169312163
      },
      "wall_time_ms_mean": 2006.6748795714896,
      "world_model_forwards": 17010
    },
    "mpc_rrt_ensemble_unc/17": {
      "abstain_rate": 0.0,
      "beam_nodes": 3402,
      "both_empty_fraction": 0.47619047619047616,
      "candidate_recall": 0.9365079365079365,
      "catalog_coverage": 0.9841269841269841,
      "conditional_on_recommendation": {
        "f1": 0.5346938775510204,
        "jaccard": 0.5177248677248677,
        "precision": 0.521164021164021,
        "recall": 0.5608465608465608
      },
      "interpretation": "agreement with recorded treatment; not causal treatment quality",
      "n": 63,
      "plan_revision_evaluable_count": 41,
      "plan_revision_evaluable_fraction": 0.6507936507936508,
      "plan_revision_interpretation": "behavioral response to new observations; a higher rate is not inherently better",
      "plan_revision_rate": 0.2926829268292683,
      "plan_revision_reason": null,
      "policy_calls": 0,
      "recommendation_coverage": 1.0,
      "recommendations": 63,
      "sequential_action_change_count": 41,
      "sequential_action_change_rate": 0.1951219512195122,
      "structural_rule_violation_rate": null,
      "structural_rule_violation_reason": "clinical_rules_disabled",
      "unconditional": {
        "f1": 0.5346938775510204,
        "jaccard": 0.5177248677248677,
        "precision": 0.521164021164021,
        "recall": 0.5608465608465608
      },
      "wall_time_ms_mean": 1623.6398336625407,
      "world_model_forwards": 17010
    },
    "mpc_rrt_ensemble_unc/29": {
      "abstain_rate": 0.0,
      "beam_nodes": 3402,
      "both_empty_fraction": 0.2698412698412698,
      "candidate_recall": 0.9365079365079365,
      "catalog_coverage": 0.9841269841269841,
      "conditional_on_recommendation": {
        "f1": 0.3507936507936508,
        "jaccard": 0.3412698412698413,
        "precision": 0.35714285714285715,
        "recall": 0.3465608465608466
      },
      "interpretation": "agreement with recorded treatment; not causal treatment quality",
      "n": 63,
      "plan_revision_evaluable_count": 41,
      "plan_revision_evaluable_fraction": 0.6507936507936508,
      "plan_revision_interpretation": "behavioral response to new observations; a higher rate is not inherently better",
      "plan_revision_rate": 0.4146341463414634,
      "plan_revision_reason": null,
      "policy_calls": 0,
      "recommendation_coverage": 1.0,
      "recommendations": 63,
      "sequential_action_change_count": 41,
      "sequential_action_change_rate": 0.2926829268292683,
      "structural_rule_violation_rate": null,
      "structural_rule_violation_reason": "clinical_rules_disabled",
      "unconditional": {
        "f1": 0.3507936507936508,
        "jaccard": 0.3412698412698413,
        "precision": 0.35714285714285715,
        "recall": 0.3465608465608466
      },
      "wall_time_ms_mean": 1706.7883309819513,
      "world_model_forwards": 17010
    },
    "mpc_rrt_ensemble_unc/7": {
      "abstain_rate": 0.0,
      "beam_nodes": 3402,
      "both_empty_fraction": 0.2698412698412698,
      "candidate_recall": 0.9365079365079365,
      "catalog_coverage": 0.9841269841269841,
      "conditional_on_recommendation": {
        "f1": 0.35910808767951624,
        "jaccard": 0.34259259259259256,
        "precision": 0.3544973544973545,
        "recall": 0.37169312169312163
      },
      "interpretation": "agreement with recorded treatment; not causal treatment quality",
      "n": 63,
      "plan_revision_evaluable_count": 41,
      "plan_revision_evaluable_fraction": 0.6507936507936508,
      "plan_revision_interpretation": "behavioral response to new observations; a higher rate is not inherently better",
      "plan_revision_rate": 0.36585365853658536,
      "plan_revision_reason": null,
      "policy_calls": 0,
      "recommendation_coverage": 1.0,
      "recommendations": 63,
      "sequential_action_change_count": 41,
      "sequential_action_change_rate": 0.3170731707317073,
      "structural_rule_violation_rate": null,
      "structural_rule_violation_reason": "clinical_rules_disabled",
      "unconditional": {
        "f1": 0.35910808767951624,
        "jaccard": 0.34259259259259256,
        "precision": 0.3544973544973545,
        "recall": 0.37169312169312163
      },
      "wall_time_ms_mean": 1940.7554126006462,
      "world_model_forwards": 17010
    }
  },
  "validation": {
    "fixed_plan/17": {
      "abstain_rate": 0.0,
      "beam_nodes": 3348,
      "both_empty_fraction": 0.12903225806451613,
      "candidate_recall": 0.9032258064516129,
      "catalog_coverage": 0.9354838709677419,
      "conditional_on_recommendation": {
        "f1": 0.3044546850998464,
        "jaccard": 0.2639784946236559,
        "precision": 0.28225806451612895,
        "recall": 0.3481182795698925
      },
      "interpretation": "agreement with recorded treatment; not causal treatment quality",
      "n": 62,
      "plan_revision_evaluable_count": 0,
      "plan_revision_evaluable_fraction": 0.0,
      "plan_revision_interpretation": "behavioral response to new observations; a higher rate is not inherently better",
      "plan_revision_rate": null,
      "plan_revision_reason": "fixed_plan_does_not_replan",
      "policy_calls": 0,
      "recommendation_coverage": 1.0,
      "recommendations": 62,
      "sequential_action_change_count": 39,
      "sequential_action_change_rate": 0.3076923076923077,
      "structural_rule_violation_rate": null,
      "structural_rule_violation_reason": "clinical_rules_disabled",
      "unconditional": {
        "f1": 0.3044546850998464,
        "jaccard": 0.2639784946236559,
        "precision": 0.28225806451612895,
        "recall": 0.3481182795698925
      },
      "wall_time_ms_mean": 1985.4648194525525,
      "world_model_forwards": 16740
    },
    "fixed_plan/29": {
      "abstain_rate": 0.0,
      "beam_nodes": 3348,
      "both_empty_fraction": 0.08064516129032258,
      "candidate_recall": 0.9032258064516129,
      "catalog_coverage": 0.9354838709677419,
      "conditional_on_recommendation": {
        "f1": 0.2533026113671275,
        "jaccard": 0.22634408602150538,
        "precision": 0.26075268817204295,
        "recall": 0.2647849462365592
      },
      "interpretation": "agreement with recorded treatment; not causal treatment quality",
      "n": 62,
      "plan_revision_evaluable_count": 0,
      "plan_revision_evaluable_fraction": 0.0,
      "plan_revision_interpretation": "behavioral response to new observations; a higher rate is not inherently better",
      "plan_revision_rate": null,
      "plan_revision_reason": "fixed_plan_does_not_replan",
      "policy_calls": 0,
      "recommendation_coverage": 1.0,
      "recommendations": 62,
      "sequential_action_change_count": 39,
      "sequential_action_change_rate": 0.48717948717948717,
      "structural_rule_violation_rate": null,
      "structural_rule_violation_reason": "clinical_rules_disabled",
      "unconditional": {
        "f1": 0.2533026113671275,
        "jaccard": 0.22634408602150538,
        "precision": 0.26075268817204295,
        "recall": 0.2647849462365592
      },
      "wall_time_ms_mean": 2010.640064693777,
      "world_model_forwards": 16740
    },
    "fixed_plan/7": {
      "abstain_rate": 0.0,
      "beam_nodes": 3348,
      "both_empty_fraction": 0.11290322580645161,
      "candidate_recall": 0.9032258064516129,
      "catalog_coverage": 0.9354838709677419,
      "conditional_on_recommendation": {
        "f1": 0.2588325652841782,
        "jaccard": 0.21774193548387097,
        "precision": 0.24731182795698925,
        "recall": 0.28225806451612906
      },
      "interpretation": "agreement with recorded treatment; not causal treatment quality",
      "n": 62,
      "plan_revision_evaluable_count": 0,
      "plan_revision_evaluable_fraction": 0.0,
      "plan_revision_interpretation": "behavioral response to new observations; a higher rate is not inherently better",
      "plan_revision_rate": null,
      "plan_revision_reason": "fixed_plan_does_not_replan",
      "policy_calls": 0,
      "recommendation_coverage": 1.0,
      "recommendations": 62,
      "sequential_action_change_count": 39,
      "sequential_action_change_rate": 0.28205128205128205,
      "structural_rule_violation_rate": null,
      "structural_rule_violation_reason": "clinical_rules_disabled",
      "unconditional": {
        "f1": 0.2588325652841782,
        "jaccard": 0.21774193548387097,
        "precision": 0.24731182795698925,
        "recall": 0.28225806451612906
      },
      "wall_time_ms_mean": 2104.0358039204993,
      "world_model_forwards": 16740
    },
    "frequency/17": {
      "abstain_rate": 0.0,
      "beam_nodes": 0,
      "both_empty_fraction": 0.3870967741935484,
      "candidate_recall": 0.9032258064516129,
      "catalog_coverage": 0.9354838709677419,
      "conditional_on_recommendation": {
        "f1": 0.3870967741935484,
        "jaccard": 0.3870967741935484,
        "precision": 0.3870967741935484,
        "recall": 0.3870967741935484
      },
      "interpretation": "agreement with recorded treatment; not causal treatment quality",
      "n": 62,
      "plan_revision_evaluable_count": 0,
      "plan_revision_evaluable_fraction": 0.0,
      "plan_revision_interpretation": "behavioral response to new observations; a higher rate is not inherently better",
      "plan_revision_rate": null,
      "plan_revision_reason": "no_prior_future_action_in_h1_policy",
      "policy_calls": 0,
      "recommendation_coverage": 1.0,
      "recommendations": 62,
      "sequential_action_change_count": 39,
      "sequential_action_change_rate": 0.0,
      "structural_rule_violation_rate": null,
      "structural_rule_violation_reason": "clinical_rules_disabled",
      "unconditional": {
        "f1": 0.3870967741935484,
        "jaccard": 0.3870967741935484,
        "precision": 0.3870967741935484,
        "recall": 0.3870967741935484
      },
      "wall_time_ms_mean": 0.24856607084192575,
      "world_model_forwards": 0
    },
    "frequency/29": {
      "abstain_rate": 0.0,
      "beam_nodes": 0,
      "both_empty_fraction": 0.3870967741935484,
      "candidate_recall": 0.9032258064516129,
      "catalog_coverage": 0.9354838709677419,
      "conditional_on_recommendation": {
        "f1": 0.3870967741935484,
        "jaccard": 0.3870967741935484,
        "precision": 0.3870967741935484,
        "recall": 0.3870967741935484
      },
      "interpretation": "agreement with recorded treatment; not causal treatment quality",
      "n": 62,
      "plan_revision_evaluable_count": 0,
      "plan_revision_evaluable_fraction": 0.0,
      "plan_revision_interpretation": "behavioral response to new observations; a higher rate is not inherently better",
      "plan_revision_rate": null,
      "plan_revision_reason": "no_prior_future_action_in_h1_policy",
      "policy_calls": 0,
      "recommendation_coverage": 1.0,
      "recommendations": 62,
      "sequential_action_change_count": 39,
      "sequential_action_change_rate": 0.0,
      "structural_rule_violation_rate": null,
      "structural_rule_violation_reason": "clinical_rules_disabled",
      "unconditional": {
        "f1": 0.3870967741935484,
        "jaccard": 0.3870967741935484,
        "precision": 0.3870967741935484,
        "recall": 0.3870967741935484
      },
      "wall_time_ms_mean": 0.27277629030117345,
      "world_model_forwards": 0
    },
    "frequency/7": {
      "abstain_rate": 0.0,
      "beam_nodes": 0,
      "both_empty_fraction": 0.3870967741935484,
      "candidate_recall": 0.9032258064516129,
      "catalog_coverage": 0.9354838709677419,
      "conditional_on_recommendation": {
        "f1": 0.3870967741935484,
        "jaccard": 0.3870967741935484,
        "precision": 0.3870967741935484,
        "recall": 0.3870967741935484
      },
      "interpretation": "agreement with recorded treatment; not causal treatment quality",
      "n": 62,
      "plan_revision_evaluable_count": 0,
      "plan_revision_evaluable_fraction": 0.0,
      "plan_revision_interpretation": "behavioral response to new observations; a higher rate is not inherently better",
      "plan_revision_rate": null,
      "plan_revision_reason": "no_prior_future_action_in_h1_policy",
      "policy_calls": 0,
      "recommendation_coverage": 1.0,
      "recommendations": 62,
      "sequential_action_change_count": 39,
      "sequential_action_change_rate": 0.0,
      "structural_rule_violation_rate": null,
      "structural_rule_violation_reason": "clinical_rules_disabled",
      "unconditional": {
        "f1": 0.3870967741935484,
        "jaccard": 0.3870967741935484,
        "precision": 0.3870967741935484,
        "recall": 0.3870967741935484
      },
      "wall_time_ms_mean": 0.18668871076266852,
      "world_model_forwards": 0
    },
    "greedy/17": {
      "abstain_rate": 0.0,
      "beam_nodes": 372,
      "both_empty_fraction": 0.0,
      "candidate_recall": 0.9032258064516129,
      "catalog_coverage": 0.9354838709677419,
      "conditional_on_recommendation": {
        "f1": 0.33118279569892467,
        "jaccard": 0.26801075268817204,
        "precision": 0.29301075268817195,
        "recall": 0.4086021505376344
      },
      "interpretation": "agreement with recorded treatment; not causal treatment quality",
      "n": 62,
      "plan_revision_evaluable_count": 0,
      "plan_revision_evaluable_fraction": 0.0,
      "plan_revision_interpretation": "behavioral response to new observations; a higher rate is not inherently better",
      "plan_revision_rate": null,
      "plan_revision_reason": "no_prior_future_action_in_h1_policy",
      "policy_calls": 0,
      "recommendation_coverage": 1.0,
      "recommendations": 62,
      "sequential_action_change_count": 39,
      "sequential_action_change_rate": 0.23076923076923078,
      "structural_rule_violation_rate": null,
      "structural_rule_violation_reason": "clinical_rules_disabled",
      "unconditional": {
        "f1": 0.33118279569892467,
        "jaccard": 0.26801075268817204,
        "precision": 0.29301075268817195,
        "recall": 0.4086021505376344
      },
      "wall_time_ms_mean": 232.37748990105766,
      "world_model_forwards": 1860
    },
    "greedy/29": {
      "abstain_rate": 0.0,
      "beam_nodes": 372,
      "both_empty_fraction": 0.11290322580645161,
      "candidate_recall": 0.9032258064516129,
      "catalog_coverage": 0.9354838709677419,
      "conditional_on_recommendation": {
        "f1": 0.3366359447004608,
        "jaccard": 0.2881720430107527,
        "precision": 0.31586021505376344,
        "recall": 0.3951612903225806
      },
      "interpretation": "agreement with recorded treatment; not causal treatment quality",
      "n": 62,
      "plan_revision_evaluable_count": 0,
      "plan_revision_evaluable_fraction": 0.0,
      "plan_revision_interpretation": "behavioral response to new observations; a higher rate is not inherently better",
      "plan_revision_rate": null,
      "plan_revision_reason": "no_prior_future_action_in_h1_policy",
      "policy_calls": 0,
      "recommendation_coverage": 1.0,
      "recommendations": 62,
      "sequential_action_change_count": 39,
      "sequential_action_change_rate": 0.48717948717948717,
      "structural_rule_violation_rate": null,
      "structural_rule_violation_reason": "clinical_rules_disabled",
      "unconditional": {
        "f1": 0.3366359447004608,
        "jaccard": 0.2881720430107527,
        "precision": 0.31586021505376344,
        "recall": 0.3951612903225806
      },
      "wall_time_ms_mean": 226.36557094876716,
      "world_model_forwards": 1860
    },
    "greedy/7": {
      "abstain_rate": 0.0,
      "beam_nodes": 372,
      "both_empty_fraction": 0.14516129032258066,
      "candidate_recall": 0.9032258064516129,
      "catalog_coverage": 0.9354838709677419,
      "conditional_on_recommendation": {
        "f1": 0.31044546850998456,
        "jaccard": 0.2693548387096774,
        "precision": 0.2956989247311827,
        "recall": 0.3400537634408602
      },
      "interpretation": "agreement with recorded treatment; not causal treatment quality",
      "n": 62,
      "plan_revision_evaluable_count": 0,
      "plan_revision_evaluable_fraction": 0.0,
      "plan_revision_interpretation": "behavioral response to new observations; a higher rate is not inherently better",
      "plan_revision_rate": null,
      "plan_revision_reason": "no_prior_future_action_in_h1_policy",
      "policy_calls": 0,
      "recommendation_coverage": 1.0,
      "recommendations": 62,
      "sequential_action_change_count": 39,
      "sequential_action_change_rate": 0.4358974358974359,
      "structural_rule_violation_rate": null,
      "structural_rule_violation_reason": "clinical_rules_disabled",
      "unconditional": {
        "f1": 0.31044546850998456,
        "jaccard": 0.2693548387096774,
        "precision": 0.2956989247311827,
        "recall": 0.3400537634408602
      },
      "wall_time_ms_mean": 306.8277144216887,
      "world_model_forwards": 1860
    },
    "mpc_ensemble/17": {
      "abstain_rate": 0.0,
      "beam_nodes": 3348,
      "both_empty_fraction": 0.27419354838709675,
      "candidate_recall": 0.9032258064516129,
      "catalog_coverage": 0.9354838709677419,
      "conditional_on_recommendation": {
        "f1": 0.37342549923195084,
        "jaccard": 0.353494623655914,
        "precision": 0.3602150537634409,
        "recall": 0.39919354838709675
      },
      "interpretation": "agreement with recorded treatment; not causal treatment quality",
      "n": 62,
      "plan_revision_evaluable_count": 39,
      "plan_revision_evaluable_fraction": 0.6290322580645161,
      "plan_revision_interpretation": "behavioral response to new observations; a higher rate is not inherently better",
      "plan_revision_rate": 0.46153846153846156,
      "plan_revision_reason": null,
      "policy_calls": 0,
      "recommendation_coverage": 1.0,
      "recommendations": 62,
      "sequential_action_change_count": 39,
      "sequential_action_change_rate": 0.3076923076923077,
      "structural_rule_violation_rate": null,
      "structural_rule_violation_reason": "clinical_rules_disabled",
      "unconditional": {
        "f1": 0.37342549923195084,
        "jaccard": 0.353494623655914,
        "precision": 0.3602150537634409,
        "recall": 0.39919354838709675
      },
      "wall_time_ms_mean": 1986.2575681632266,
      "world_model_forwards": 16740
    },
    "mpc_ensemble/29": {
      "abstain_rate": 0.0,
      "beam_nodes": 3348,
      "both_empty_fraction": 0.16129032258064516,
      "candidate_recall": 0.9032258064516129,
      "catalog_coverage": 0.9354838709677419,
      "conditional_on_recommendation": {
        "f1": 0.3199692780337941,
        "jaccard": 0.29569892473118276,
        "precision": 0.3400537634408602,
        "recall": 0.3131720430107527
      },
      "interpretation": "agreement with recorded treatment; not causal treatment quality",
      "n": 62,
      "plan_revision_evaluable_count": 39,
      "plan_revision_evaluable_fraction": 0.6290322580645161,
      "plan_revision_interpretation": "behavioral response to new observations; a higher rate is not inherently better",
      "plan_revision_rate": 0.4358974358974359,
      "plan_revision_reason": null,
      "policy_calls": 0,
      "recommendation_coverage": 1.0,
      "recommendations": 62,
      "sequential_action_change_count": 39,
      "sequential_action_change_rate": 0.358974358974359,
      "structural_rule_violation_rate": null,
      "structural_rule_violation_reason": "clinical_rules_disabled",
      "unconditional": {
        "f1": 0.3199692780337941,
        "jaccard": 0.29569892473118276,
        "precision": 0.3400537634408602,
        "recall": 0.3131720430107527
      },
      "wall_time_ms_mean": 1991.2997076612703,
      "world_model_forwards": 16740
    },
    "mpc_ensemble/7": {
      "abstain_rate": 0.0,
      "beam_nodes": 3348,
      "both_empty_fraction": 0.14516129032258066,
      "candidate_recall": 0.9032258064516129,
      "catalog_coverage": 0.9354838709677419,
      "conditional_on_recommendation": {
        "f1": 0.26927803379416276,
        "jaccard": 0.23521505376344085,
        "precision": 0.26075268817204295,
        "recall": 0.2862903225806452
      },
      "interpretation": "agreement with recorded treatment; not causal treatment quality",
      "n": 62,
      "plan_revision_evaluable_count": 39,
      "plan_revision_evaluable_fraction": 0.6290322580645161,
      "plan_revision_interpretation": "behavioral response to new observations; a higher rate is not inherently better",
      "plan_revision_rate": 0.20512820512820512,
      "plan_revision_reason": null,
      "policy_calls": 0,
      "recommendation_coverage": 1.0,
      "recommendations": 62,
      "sequential_action_change_count": 39,
      "sequential_action_change_rate": 0.23076923076923078,
      "structural_rule_violation_rate": null,
      "structural_rule_violation_reason": "clinical_rules_disabled",
      "unconditional": {
        "f1": 0.26927803379416276,
        "jaccard": 0.23521505376344085,
        "precision": 0.26075268817204295,
        "recall": 0.2862903225806452
      },
      "wall_time_ms_mean": 2175.518901209392,
      "world_model_forwards": 16740
    },
    "mpc_rrt_ensemble/17": {
      "abstain_rate": 0.0,
      "beam_nodes": 3348,
      "both_empty_fraction": 0.20967741935483872,
      "candidate_recall": 0.9032258064516129,
      "catalog_coverage": 0.9354838709677419,
      "conditional_on_recommendation": {
        "f1": 0.33932411674347157,
        "jaccard": 0.31505376344086017,
        "precision": 0.3225806451612903,
        "recall": 0.3723118279569893
      },
      "interpretation": "agreement with recorded treatment; not causal treatment quality",
      "n": 62,
      "plan_revision_evaluable_count": 39,
      "plan_revision_evaluable_fraction": 0.6290322580645161,
      "plan_revision_interpretation": "behavioral response to new observations; a higher rate is not inherently better",
      "plan_revision_rate": 0.4358974358974359,
      "plan_revision_reason": null,
      "policy_calls": 0,
      "recommendation_coverage": 1.0,
      "recommendations": 62,
      "sequential_action_change_count": 39,
      "sequential_action_change_rate": 0.46153846153846156,
      "structural_rule_violation_rate": null,
      "structural_rule_violation_reason": "clinical_rules_disabled",
      "unconditional": {
        "f1": 0.33932411674347157,
        "jaccard": 0.31505376344086017,
        "precision": 0.3225806451612903,
        "recall": 0.3723118279569893
      },
      "wall_time_ms_mean": 2000.810104400678,
      "world_model_forwards": 16740
    },
    "mpc_rrt_ensemble/29": {
      "abstain_rate": 0.0,
      "beam_nodes": 3348,
      "both_empty_fraction": 0.14516129032258066,
      "candidate_recall": 0.9032258064516129,
      "catalog_coverage": 0.9354838709677419,
      "conditional_on_recommendation": {
        "f1": 0.30921658986175116,
        "jaccard": 0.2876344086021505,
        "precision": 0.3239247311827957,
        "recall": 0.30510752688172044
      },
      "interpretation": "agreement with recorded treatment; not causal treatment quality",
      "n": 62,
      "plan_revision_evaluable_count": 39,
      "plan_revision_evaluable_fraction": 0.6290322580645161,
      "plan_revision_interpretation": "behavioral response to new observations; a higher rate is not inherently better",
      "plan_revision_rate": 0.5128205128205128,
      "plan_revision_reason": null,
      "policy_calls": 0,
      "recommendation_coverage": 1.0,
      "recommendations": 62,
      "sequential_action_change_count": 39,
      "sequential_action_change_rate": 0.48717948717948717,
      "structural_rule_violation_rate": null,
      "structural_rule_violation_reason": "clinical_rules_disabled",
      "unconditional": {
        "f1": 0.30921658986175116,
        "jaccard": 0.2876344086021505,
        "precision": 0.3239247311827957,
        "recall": 0.30510752688172044
      },
      "wall_time_ms_mean": 1985.9760733858323,
      "world_model_forwards": 16740
    },
    "mpc_rrt_ensemble/7": {
      "abstain_rate": 0.0,
      "beam_nodes": 3348,
      "both_empty_fraction": 0.1774193548387097,
      "candidate_recall": 0.9032258064516129,
      "catalog_coverage": 0.9354838709677419,
      "conditional_on_recommendation": {
        "f1": 0.3233486943164362,
        "jaccard": 0.28225806451612906,
        "precision": 0.31182795698924726,
        "recall": 0.3467741935483871
      },
      "interpretation": "agreement with recorded treatment; not causal treatment quality",
      "n": 62,
      "plan_revision_evaluable_count": 39,
      "plan_revision_evaluable_fraction": 0.6290322580645161,
      "plan_revision_interpretation": "behavioral response to new observations; a higher rate is not inherently better",
      "plan_revision_rate": 0.3333333333333333,
      "plan_revision_reason": null,
      "policy_calls": 0,
      "recommendation_coverage": 1.0,
      "recommendations": 62,
      "sequential_action_change_count": 39,
      "sequential_action_change_rate": 0.3076923076923077,
      "structural_rule_violation_rate": null,
      "structural_rule_violation_reason": "clinical_rules_disabled",
      "unconditional": {
        "f1": 0.3233486943164362,
        "jaccard": 0.28225806451612906,
        "precision": 0.31182795698924726,
        "recall": 0.3467741935483871
      },
      "wall_time_ms_mean": 1935.7376510320578,
      "world_model_forwards": 16740
    },
    "mpc_rrt_ensemble_unc/17": {
      "abstain_rate": 0.0,
      "beam_nodes": 3348,
      "both_empty_fraction": 0.20967741935483872,
      "candidate_recall": 0.9032258064516129,
      "catalog_coverage": 0.9354838709677419,
      "conditional_on_recommendation": {
        "f1": 0.33932411674347157,
        "jaccard": 0.31505376344086017,
        "precision": 0.3225806451612903,
        "recall": 0.3723118279569893
      },
      "interpretation": "agreement with recorded treatment; not causal treatment quality",
      "n": 62,
      "plan_revision_evaluable_count": 39,
      "plan_revision_evaluable_fraction": 0.6290322580645161,
      "plan_revision_interpretation": "behavioral response to new observations; a higher rate is not inherently better",
      "plan_revision_rate": 0.4358974358974359,
      "plan_revision_reason": null,
      "policy_calls": 0,
      "recommendation_coverage": 1.0,
      "recommendations": 62,
      "sequential_action_change_count": 39,
      "sequential_action_change_rate": 0.46153846153846156,
      "structural_rule_violation_rate": null,
      "structural_rule_violation_reason": "clinical_rules_disabled",
      "unconditional": {
        "f1": 0.33932411674347157,
        "jaccard": 0.31505376344086017,
        "precision": 0.3225806451612903,
        "recall": 0.3723118279569893
      },
      "wall_time_ms_mean": 1992.0525784874635,
      "world_model_forwards": 16740
    },
    "mpc_rrt_ensemble_unc/29": {
      "abstain_rate": 0.0,
      "beam_nodes": 3348,
      "both_empty_fraction": 0.14516129032258066,
      "candidate_recall": 0.9032258064516129,
      "catalog_coverage": 0.9354838709677419,
      "conditional_on_recommendation": {
        "f1": 0.34147465437788016,
        "jaccard": 0.31989247311827956,
        "precision": 0.3561827956989247,
        "recall": 0.3373655913978495
      },
      "interpretation": "agreement with recorded treatment; not causal treatment quality",
      "n": 62,
      "plan_revision_evaluable_count": 39,
      "plan_revision_evaluable_fraction": 0.6290322580645161,
      "plan_revision_interpretation": "behavioral response to new observations; a higher rate is not inherently better",
      "plan_revision_rate": 0.5128205128205128,
      "plan_revision_reason": null,
      "policy_calls": 0,
      "recommendation_coverage": 1.0,
      "recommendations": 62,
      "sequential_action_change_count": 39,
      "sequential_action_change_rate": 0.48717948717948717,
      "structural_rule_violation_rate": null,
      "structural_rule_violation_reason": "clinical_rules_disabled",
      "unconditional": {
        "f1": 0.34147465437788016,
        "jaccard": 0.31989247311827956,
        "precision": 0.3561827956989247,
        "recall": 0.3373655913978495
      },
      "wall_time_ms_mean": 1997.9218423550558,
      "world_model_forwards": 16740
    },
    "mpc_rrt_ensemble_unc/7": {
      "abstain_rate": 0.0,
      "beam_nodes": 3348,
      "both_empty_fraction": 0.1774193548387097,
      "candidate_recall": 0.9032258064516129,
      "catalog_coverage": 0.9354838709677419,
      "conditional_on_recommendation": {
        "f1": 0.31182795698924726,
        "jaccard": 0.26881720430107525,
        "precision": 0.2997311827956988,
        "recall": 0.33602150537634407
      },
      "interpretation": "agreement with recorded treatment; not causal treatment quality",
      "n": 62,
      "plan_revision_evaluable_count": 39,
      "plan_revision_evaluable_fraction": 0.6290322580645161,
      "plan_revision_interpretation": "behavioral response to new observations; a higher rate is not inherently better",
      "plan_revision_rate": 0.3333333333333333,
      "plan_revision_reason": null,
      "policy_calls": 0,
      "recommendation_coverage": 1.0,
      "recommendations": 62,
      "sequential_action_change_count": 39,
      "sequential_action_change_rate": 0.3076923076923077,
      "structural_rule_violation_rate": null,
      "structural_rule_violation_reason": "clinical_rules_disabled",
      "unconditional": {
        "f1": 0.31182795698924726,
        "jaccard": 0.26881720430107525,
        "precision": 0.2997311827956988,
        "recall": 0.33602150537634407
      },
      "wall_time_ms_mean": 1921.9475905579936,
      "world_model_forwards": 16740
    }
  }
}
```

## Independent synthetic environment

Not completed.

## Resource and artifact summary

- `metrics.json`: 366060 bytes
- `models.pt`: 59700495 bytes
- `predictions.jsonl`: 4399042 bytes
- `run.json`: 23750 bytes

## Interpretation limits

- planning cost is an observational prognostic proxy, not a causal treatment effect
- historical replay evaluates agreement and behavior, not counterfactual benefit
- synthetic closed-loop results do not establish clinical efficacy
- D1/D2 are internal controlled comparisons, not the full official CLARITY baseline.
- Historical next MRI observations are never scored as outcomes of a different recommended action.
- Synthetic environment performance must not be presented as patient survival benefit.

## Incomplete items

None.
