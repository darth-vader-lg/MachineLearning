using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms.Text;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace MachineLearning
{
   /// <summary>
   /// Classe per l'interpretazione del significato si testi
   /// </summary>
   [Serializable]
   public sealed partial class PredictorTextMeaning : Predictor, IDataStorageProvider, IModelStorageProvider, ITextDataOptionsProvider, ITrainingDataProvider
   {
      #region Fields
      /// <summary>
      /// Nome colonna label
      /// </summary>
      private string _labelColumnName;
      /// <summary>
      /// Pipe di training
      /// </summary>
      [NonSerialized]
      private IEstimator<ITransformer> _pipe;
      /// <summary>
      /// Contatore di retrain
      /// </summary>
      [NonSerialized]
      private int _retrainCount;
      #endregion
      #region Properties
      /// <summary>
      /// Storage dei dati
      /// </summary>
      public IDataStorage DataStorage { get; set; }
      /// <summary>
      /// Numero massimo di tentativi di retrain del modello
      /// </summary>
      public int MaxRetrain { get; set; } = 1;
      /// <summary>
      /// Storage del modello
      /// </summary>
      public IModelStorage ModelStorage { get; set; }
      /// <summary>
      /// Pipe di training
      /// </summary>
      private IEstimator<ITransformer> Pipe { get => _pipe; set => _pipe = value; }
      /// <summary>
      /// Seme per le operazioni random
      /// </summary>
      private int Seed { get; set; }
      /// <summary>
      /// Opzioni di caricamento dati testuali
      /// </summary>
      public TextDataOptions TextDataOptions { get; private set; }
      /// <summary>
      /// Dati di training
      /// </summary>
      public ITrainingData TrainingData { get; set; }
      #endregion
      #region Methods
      /// <summary>
      /// Costruttore
      /// </summary>
      public PredictorTextMeaning() => Init();
      /// <summary>
      /// Costruttore
      /// </summary>
      /// <param name="seed">Seme operazioni random</param>
      public PredictorTextMeaning(int? seed) : base(seed) => Init();
      /// <summary>
      /// Costruttore
      /// </summary>
      /// <param name="ml">Contesto di machine learning</param>
      public PredictorTextMeaning(MachineLearningContext ml) : base(ml) => Init();
      /// <summary>
      /// Funzione di inizializzazione
      /// </summary>
      private void Init() => SetDataFormat("Label", new TextDataOptions { AllowQuoting = true, Separators = new[] { ',' } });
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
      /// Restituisce la previsione
      /// </summary>
      /// <param name="sentence">Significato da prevedere</param>
      /// <returns>La previsione</returns>
      public Prediction GetPrediction(string sentence) => new Prediction(GetPredictionData(null, sentence));
      /// <summary>
      /// Restituisce la previsione
      /// </summary>
      /// <param name="sentence">Significato da prevedere</param>
      /// <returns>Il task della previsione</returns>
      public async Task<Prediction> GetPredictionAsync(string sentence, CancellationToken cancel = default) => new Prediction(await GetPredictionDataAsync(cancel, null, sentence));
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
      /// Restituisce il modello effettuando il training. Da implementare nelle classi derivate
      /// </summary>
      /// <param name="dataView">Datidi training</param>
      /// <param name="cancellation">Token di annullamento</param>
      /// <returns>Il modello appreso</returns>
      protected override ITransformer GetTrainedModel(IDataView dataView, CancellationToken cancellation)
      {
         // Verifica numero di tentativi massimi di retrain raggiunto
         if (++_retrainCount > MaxRetrain)
            return null;
         // Pipe di training
         Pipe ??=
            ML.NET.Transforms.Conversion.MapValueToKey("Label", _labelColumnName).
            Append(ML.NET.Transforms.Text.FeaturizeText("FeaturizeText", new TextFeaturizingEstimator.Options(), (from c in Evaluation.InputSchema
                                                                                                                  where c.Name != _labelColumnName
                                                                                                                  select c.Name).ToArray())).
            Append(ML.NET.Transforms.CopyColumns("Features", "FeaturizeText")).
            Append(ML.NET.Transforms.NormalizeMinMax("Features")).
            AppendCacheCheckpoint(ML.NET).
            Append(ML.NET.MulticlassClassification.Trainers.SdcaNonCalibrated()).
            Append(ML.NET.Transforms.Conversion.MapKeyToValue("PredictedLabel"));
         // Mischia le linee
         var data = ML.NET.Data.ShuffleRows(dataView, Seed++);
         return Pipe.Fit(data);
      }
      /// <summary>
      /// Imposta il formato di input dei dati
      /// </summary>
      /// <param name="labelColumnName">Nome colonna label (significato delle frasi)</param>
      /// <param name="options">Opzioni</param>
      public void SetDataFormat(string labelColumnName = "Label", TextDataOptions options = default)
      {
         _labelColumnName = !string.IsNullOrWhiteSpace(labelColumnName) ? labelColumnName : "Label";
         if (options == default)
            options = new TextDataOptions();
         if (options.Columns == null) {
            options.Columns = new TextDataOptions(new TextLoader.Options
            {
               Columns = new[]
               {
                  new TextLoader.Column(!string.IsNullOrWhiteSpace(labelColumnName) ? labelColumnName : "Label", DataKind.String, 0),
                  new TextLoader.Column("Sentence", DataKind.String, 1),
               }
            }).Columns;
         }
         else if (!options.Columns.Any(c => c.Name == _labelColumnName))
            throw new ArgumentException("Label column not defined in the input schema");
         TextDataOptions = options;
      }
      #endregion
   }

   /// <summary>
   /// Risultato della previsione
   /// </summary>
   public sealed partial class PredictorTextMeaning // Prediction
   {
      [Serializable]
      public class Prediction
      {
         #region Properties
         /// <summary>
         /// Significato
         /// </summary>
         public string Meaning { get; }
         /// <summary>
         /// Punteggi per label
         /// </summary>
         public KeyValuePair<string, float>[] Scores { get; }
         #endregion
         #region Methods
         /// <summary>
         /// Costruttore
         /// </summary>
         /// <param name="data">Dati della previsione</param>
         internal Prediction(IDataView data)
         {
            Meaning = data.GetString("PredictedLabel");
            var slotNames = default(VBuffer<ReadOnlyMemory<char>>);
            data.Schema["Score"].GetSlotNames(ref slotNames);
            var scores = data.GetValue<VBuffer<float>>("Score");
            Scores = slotNames.GetValues().ToArray().Zip(scores.GetValues().ToArray()).Select(item => new KeyValuePair<string, float>(item.First.ToString(), item.Second)).ToArray();
         }
         #endregion
      }
   }
}
