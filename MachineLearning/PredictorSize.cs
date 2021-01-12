using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace MachineLearning
{
   /// <summary>
   /// Classe per la previsione delle taglie
   /// </summary>
   public sealed partial class PredictorSize : PredictorRegression, IDataStorageProvider, IModelStorageProvider, ITextLoaderOptionsProvider, ITrainingDataProvider
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
      public TextLoader.Options TextLoaderOptions { get; private set; }
      /// <summary>
      /// Dati di training
      /// </summary>
      public IDataStorage TrainingData { get; set; }
      #endregion
      #region Methods
      /// <summary>
      /// Costruttore
      /// </summary>
      public PredictorSize() => Init();
      /// <summary>
      /// Costruttore
      /// </summary>
      /// <param name="seed">Seme operazioni random</param>
      public PredictorSize(int? seed) : base(seed) => Init();
      /// <summary>
      /// Costruttore
      /// </summary>
      /// <param name="ml">Contesto di machine learning</param>
      public PredictorSize(MachineLearningContext ml) : base(ml) => Init();
      /// <summary>
      /// Funzione di inizializzazione
      /// </summary>
      private void Init() => SetInputSchema("Label", new TextLoader.Options { AllowQuoting = true, Separators = new[] { ',' } });
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
         if (modelEvaluation1 is RegressionMetrics metrics1 && modelEvaluation2 is RegressionMetrics metrics2)
            best = metrics2.RSquared > metrics1.RSquared ? modelEvaluation2 : modelEvaluation1;
         if (best == modelEvaluation2)
            _retrainCount = 0;
         return best;
      }
      /// <summary>
      /// Restituisce la previsione
      /// </summary>
      /// <param name="values">Valori di input</param>
      /// <returns>La previsione</returns>
      public Prediction GetPrediction(params float[] values)
      {
         var nfi = new NumberFormatInfo { NumberDecimalSeparator = char.ToString(TextLoaderOptions.DecimalMarker), NumberGroupSeparator = "" };
         var data = new List<string>();
         for (var i = 0; i < values.Length; i++) {
            if (TextLoaderOptions.Columns[i].Name == _labelColumnName)
               data.Add(null);
            data.Add(values[i].ToString(nfi));
         }
         return new Prediction(this, GetPredictionData(data));
      }
      /// <summary>
      /// Restituisce la previsione
      /// </summary>
      /// <param name="sentence">Significato da prevedere</param>
      /// <returns>Il task della previsione</returns>
      public async Task<Prediction> GetPredictionAsync(CancellationToken cancel = default, params float[] values)
      {
         var nfi = new NumberFormatInfo { NumberDecimalSeparator = char.ToString(TextLoaderOptions.DecimalMarker), NumberGroupSeparator = "" };
         var data = new List<string>();
         for (var i = 0; i < values.Length; i++) {
            if (TextLoaderOptions.Columns[i].Name == _labelColumnName)
               data.Add(null);
            data.Add(values[i].ToString(nfi));
         }
         return new Prediction(this, await GetPredictionDataAsync(data, cancel));
      }
      /// <summary>
      /// Funzione di restituzione della valutazione del modello (metrica, accuratezza, ecc...)
      /// </summary>
      /// <param name="model">Modello da valutare</param>
      /// <param name="data">Dati attuali caricati</param>
      /// <returns>Il risultato della valutazione</returns>
      /// <remarks>La valutazione ottenuta verra' infine passata alla GetBestEvaluation per compaare e selezionare il modello migliore</remarks>
      protected override object GetModelEvaluation(ITransformer model, IDataView data) => ML.NET.Regression.Evaluate(model.Transform(data));
      /// <summary>
      /// Funzione di restituzione della valutazione del modello (metrica, accuratezza, ecc...)
      /// </summary>
      /// <param name="modelEvaluation">Il risultato della valutazione di un modello</param>
      /// <returns>Il risultato della valutazione in formato testo</returns>
      protected override string GetModelEvaluationInfo(object modelEvaluation)
      {
         if (modelEvaluation is not RegressionMetrics metrics)
            return null;
         var sb = new StringBuilder();
         sb.AppendLine(metrics.ToText());
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
            ML.NET.Transforms.Concatenate("Features", (from c in Evaluation.InputSchema
                                                       where c.Name != _labelColumnName
                                                       select c.Name).ToArray()).
            Append(Trainers.LightGbm(new Microsoft.ML.Trainers.LightGbm.LightGbmRegressionTrainer.Options
            {
               LabelColumnName = _labelColumnName,
            }));
         // Mischia le linee
         var data = ML.NET.Data.ShuffleRows(dataView, Seed++);
         return Pipe.Fit(data);
      }
      /// <summary>
      /// Imposta il formato di input dei dati
      /// </summary>
      /// <param name="labelColumnName">Nome colonna label (significato delle frasi)</param>
      /// <param name="options">Opzioni</param>
      public void SetInputSchema(string labelColumnName = "Label", TextLoader.Options options = default)
      {
         _labelColumnName = !string.IsNullOrWhiteSpace(labelColumnName) ? labelColumnName : "Label";
         options ??= new TextLoader.Options();
         if (options.Columns == null) {
            options.Columns = new TextLoader.Options
            {
               Columns = new[]
               {
                  new TextLoader.Column(!string.IsNullOrWhiteSpace(labelColumnName) ? labelColumnName : "Label", DataKind.Single, 0),
                  new TextLoader.Column("Data", DataKind.Single, 1),
               }
            }.Columns;
         }
         else if (!options.Columns.Any(c => c.Name == _labelColumnName))
            throw new ArgumentException("Label column not defined in the input schema");
         TextLoaderOptions = options;
      }
      #endregion
   }

   /// <summary>
   /// Algoritmo della previsione
   /// </summary>
   public sealed partial class PredictorSize // Prediction
   {
      [Serializable]
      public enum Algorithms
      {
         LbfgsMaximumEntropy,
         LightGbm,
         NaiveBayes,
         SdcaMaximumEntropy,
         SdcaNonCalibrated
      }
   }

   /// <summary>
   /// Risultato della previsione
   /// </summary>
   public sealed partial class PredictorSize // Prediction
   {
      [Serializable]
      public class Prediction
      {
         #region Properties
         /// <summary>
         /// Significato
         /// </summary>
         public float Size { get; }
         #endregion
         #region Methods
         /// <summary>
         /// Costruttore
         /// </summary>
         /// <param name="predictor">Previsore</param>
         /// <param name="data">Dati della previsione</param>
         internal Prediction(PredictorSize predictor, IDataView data)
         {
            var grid = data.ToDataViewGrid(predictor);
            Size = grid[0]["Score"];
         }
         #endregion
      }
   }
}
