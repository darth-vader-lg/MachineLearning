using MachineLearning.Data;
using MachineLearning.Model;
using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;

namespace MachineLearning
{
   /// <summary>
   /// Classe per la previsione delle taglie
   /// </summary>
   public sealed partial class SizeRecognizer :
      RegressionModelBase,
      IDataStorageProvider,
      IModelStorageProvider,
      IModelTrainerProvider,
      ITextLoaderOptionsProvider,
      ITrainingDataProvider
   {
      #region Fields
      /// <summary>
      /// Pipe di training
      /// </summary>
      [NonSerialized]
      private IEstimator<ITransformer> _pipe;
      #endregion
      #region Properties
      /// <summary>
      /// Storage dei dati
      /// </summary>
      public IDataStorage DataStorage { get; set; }
      /// <summary>
      /// Numero massimo di tentativi di training del modello
      /// </summary>
      public int MaxTrainingCycles { get; set; }
      /// <summary>
      /// Storage del modello
      /// </summary>
      public IModelStorage ModelStorage { get; set; }
      /// <summary>
      /// Trainer del modello
      /// </summary>
      public IModelTrainer ModelTrainer { get; set; }
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
      public SizeRecognizer() => Init();
      /// <summary>
      /// Costruttore
      /// </summary>
      /// <param name="seed">Seme operazioni random</param>
      public SizeRecognizer(int? seed) : base(seed) => Init();
      /// <summary>
      /// Costruttore
      /// </summary>
      /// <param name="ml">Contesto di machine learning</param>
      public SizeRecognizer(MachineLearningContext ml) : base(ml) => Init();
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
            if (TextLoaderOptions.Columns[i].Name == LabelColumnName)
               data.Add(null);
            data.Add(values[i].ToString(nfi));
         }
         return new Prediction(GetPredictionData(data));
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
            if (TextLoaderOptions.Columns[i].Name == LabelColumnName)
               data.Add(null);
            data.Add(values[i].ToString(nfi));
         }
         return new Prediction(await GetPredictionDataAsync(data, cancel));
      }
      /// <summary>
      /// Restituisce la pipe di training del modello
      /// </summary>
      /// <returns></returns>
      protected override IEstimator<ITransformer> GetPipe()
      {
         // Pipe di training
         return _pipe ??=
            ML.NET.Transforms.Concatenate("Features", (from c in Evaluation.InputSchema
                                                       where c.Name != LabelColumnName
                                                       select c.Name).ToArray()).
            Append(Trainers.LightGbm(new Microsoft.ML.Trainers.LightGbm.LightGbmRegressionTrainer.Options
            {
               LabelColumnName = LabelColumnName,
            }));
      }
      /// <summary>
      /// Funzione di inizializzazione
      /// </summary>
      private void Init() => SetInputSchema("Label", new TextLoader.Options { AllowQuoting = true, Separators = new[] { ',' } });
      /// <summary>
      /// Imposta il formato di input dei dati
      /// </summary>
      /// <param name="labelColumnName">Nome colonna label (significato delle frasi)</param>
      /// <param name="options">Opzioni</param>
      public void SetInputSchema(string labelColumnName = "Label", TextLoader.Options options = default)
      {
         LabelColumnName = !string.IsNullOrWhiteSpace(labelColumnName) ? labelColumnName : "Label";
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
         else if (!options.Columns.Any(c => c.Name == LabelColumnName))
            throw new ArgumentException("Label column not defined in the input schema");
         TextLoaderOptions = options;
      }
      #endregion
   }

   /// <summary>
   /// Algoritmo della previsione
   /// </summary>
   public sealed partial class SizeRecognizer // Prediction
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
   public sealed partial class SizeRecognizer // Prediction
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
         /// <param name="data">Dati della previsione</param>
         internal Prediction(IDataAccess data)
         {
            var grid = data.ToDataViewGrid();
            Size = grid[0]["Score"];
         }
         #endregion
      }
   }
}
