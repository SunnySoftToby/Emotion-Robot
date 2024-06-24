// /* globals Meyda */
// const levelRangeElement = document.getElementById("levelRange");
const bufferSize = 512; // 分析器的緩衝區大小
const hopSize = 256; // 分析器的跳躍大小
const numberOfMFCCCoefficients = 40; // MFCC的係數數量

let mfcc_features = [];
let allmfcc_features = [];
let recording = false; // 標記是否正在錄音
let files;
let soundurls = [];
let playingIndex = 0;

function convertToCSV(features) {
    const mfccData = features.mfcc.join(","); // 轉換為逗號分隔的字符串
    return mfccData;
}
// 開始錄音
async function startRecord(stream) {
    mfcc_features = [];
    console.log(stream)

    const audioContext = new (window.AudioContext || window.webkitAudioContext)();


    const source = audioContext.createMediaStreamSource(stream);

    const analyzer = Meyda.createMeydaAnalyzer({
        "audioContext": audioContext,
        "source": source,
        "bufferSize": bufferSize,
        "featureExtractors": ["mfcc"],
        "numberOfMFCCCoefficients": numberOfMFCCCoefficients,
        "hopSize": hopSize,
        "windowingFunction": "hanning",
        "callback": features => {
            // if (recording) {
                // const mfccCSV = convertToCSV(features);
                mfcc_features.push(features.mfcc);
                console.log(features.mfcc)
            // }
        }
    });

    analyzer.start();
    recording = true;
    console.log('取樣頻率：' + audioContext.sampleRate + ' Hz');

}
// startRecord();



// 監聽停止錄音事件
// htmlAudioElement.addEventListener("ended", function () {
//     recording = false;
//     saveToCSV();
//     startRecord();
// });


