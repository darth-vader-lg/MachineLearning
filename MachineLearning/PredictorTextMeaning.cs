using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms.Text;
using System;
using System.Linq;
using System.Text;

namespace MachineLearning
{
   /// <summary>
   /// Modello per l'interpretazione del significato si testi
   /// </summary>
   [Serializable]
   public sealed partial class PredictorTextMeaning : Predictor<string>
   {
      #region Fields
      /// <summary>
      /// Pipe di valutazione
      /// </summary>
      private IEstimator<ITransformer> _pipe;
      /// <summary>
      /// Contatore di retrain
      /// </summary>
      private int _retrainCount;
      #endregion
      #region Properties
      /// <summary>
      /// Numero massimo di tentativi di retrain del modello
      /// </summary>
      public int MaxRetrain { get; set; } = 1;
      #endregion
      #region Methods
      /// <summary>
      /// Costruttore
      /// </summary>
      public PredictorTextMeaning() => Init();
      /// <summary>
      /// Costruttore
      /// </summary>
      /// <param name="seed">Contesto di machine learning</param>
      public PredictorTextMeaning(int? seed) : base(seed) => Init();
      /// <summary>
      /// Costruttore
      /// </summary>
      /// <param name="ml">Contesto di machine learning</param>
      public PredictorTextMeaning(MachineLearningContext ml) : base(ml) => Init();
      /// <summary>
      /// Funzione di restituzione della migliore fra due valutazioni modello
      /// </summary>
      /// <param name="modelEvaluation1">Prima valutazione</param>
      /// <param name="modelEvaluation2">Seconda valutazione</param>
      /// <returns>La migliore delle due valutazioni</returns>
      /// <remarks>Tenere conto che le valutazioni potrebbero essere null</remarks>
      protected override object GetBestModelEvaluation(object modelEvaluation1, object modelEvaluation2)
      {
         var best = modelEvaluation2;
         if (modelEvaluation1 is MulticlassClassificationMetrics metrics1 && modelEvaluation2 is MulticlassClassificationMetrics metrics2)
            best = metrics2.MicroAccuracy >= metrics1.MicroAccuracy && metrics2.LogLoss < metrics1.LogLoss ? modelEvaluation2 : modelEvaluation1;
         if (best == modelEvaluation2)
            _retrainCount = 0;
         return best;
      }
      /// <summary>
      /// Funzione di restituzione della valutazione del modello (metrica, accuratezza, ecc...)
      /// </summary>
      /// <param name="model">Modello da valutare</param>
      /// <param name="data">Dati attuali caricati</param>
      /// <returns>Il risultato della valutazione</returns>
      /// <remarks>La valutazione ottenuta verra' infine passata alla GetBestEvaluation per compaare e selezionare il modello migliore</remarks>
      protected override object GetModelEvaluation(ITransformer model, IDataView data) => ML.NET.MulticlassClassification.Evaluate(model.Transform(data));
      /// <summary>
      /// Funzione di restituzione della valutazione del modello (metrica, accuratezza, ecc...)
      /// </summary>
      /// <param name="modelEvaluation">Il risultato della valutazione di un modello</param>
      /// <returns>Il risultato della valutazione in formato testo</returns>
      protected override string GetModelEvaluationInfo(object modelEvaluation)
      {
         if (modelEvaluation is not MulticlassClassificationMetrics metrics)
            return null;
         var sb = new StringBuilder();
         sb.AppendLine(metrics.ToText());
         sb.AppendLine(metrics.ConfusionMatrix.GetFormattedConfusionTable());
         return sb.ToString();
      }
      /// <summary>
      /// La pipe di stima del modello
      /// </summary>
      /// <returns>La pipe</returns>
      protected override IEstimator<ITransformer> GetPipe()
      {
         // Pipe di trasformazione
         return ++_retrainCount > MaxRetrain ? null : _pipe ??=
            ML.NET.Transforms.Conversion.MapValueToKey("Label", LabelColumnName).
            Append(ML.NET.Transforms.Text.FeaturizeText("FeaturizeText", new TextFeaturizingEstimator.Options(), (from c in Evaluation.InputSchema
                                                                                                                  where c.Name != LabelColumnName
                                                                                                                  select c.Name).ToArray())).
            Append(ML.NET.Transforms.CopyColumns("Features", "FeaturizeText")).
            Append(ML.NET.Transforms.NormalizeMinMax("Features")).
            AppendCacheCheckpoint(ML.NET).
            Append(ML.NET.MulticlassClassification.Trainers.SdcaNonCalibrated()).
            Append(ML.NET.Transforms.Conversion.MapKeyToValue(PredictionColumnName, "PredictedLabel"));
      }
      /// <summary>
      /// Funzione di notifica dello start del training
      /// </summary>
      /// <param name="e">Argomenti dell'evento</param>
      protected override void OnTrainingStarted(EventArgs e)
      {
         base.OnTrainingStarted(e);
         //@@@_retrainCount = 0;
      }
      /// <summary>
      /// Funzione di inizializzazione
      /// </summary>
      private void Init()
      {
         DataStorage = new DataStorageTextMemory();
         ModelStorage = new ModelStorageMemory();
         PredictionColumnName = "PredictedLabel";
      }
      #endregion
   }
}
