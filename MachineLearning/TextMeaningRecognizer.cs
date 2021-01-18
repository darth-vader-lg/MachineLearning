using MachineLearning.Data;
using MachineLearning.Model;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms.Text;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;

namespace MachineLearning
{
   /// <summary>
   /// Classe per l'interpretazione del significato si testi
   /// </summary>
   [Serializable]
   public sealed partial class TextMeaningRecognizer :
      MulticlassModelBase,
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
      private ModelPipes _pipes;
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
      public TextMeaningRecognizer() => Init();
      /// <summary>
      /// Costruttore
      /// </summary>
      /// <param name="seed">Seme operazioni random</param>
      public TextMeaningRecognizer(int? seed) : base(seed) => Init();
      /// <summary>
      /// Costruttore
      /// </summary>
      /// <param name="ml">Contesto di machine learning</param>
      public TextMeaningRecognizer(MachineLearningContext ml) : base(ml) => Init();
      /// <summary>
      /// Restituisce le pipe di training del modello
      /// </summary>
      /// <returns>Le pipe</returns>
      public override ModelPipes GetPipes()
      {
         // Pipe di training
         return _pipes ??= new ModelPipes
         {
            Input =
               ML.NET.Transforms.Conversion.MapValueToKey("Label", LabelColumnName)
               .Append(ML.NET.Transforms.Text.FeaturizeText("FeaturizeText", new TextFeaturizingEstimator.Options(), (from c in Evaluation.InputSchema
                                                                                                                     where c.Name != LabelColumnName
                                                                                                                     select c.Name).ToArray()))
               .Append(ML.NET.Transforms.CopyColumns("Features", "FeaturizeText"))
               .Append(ML.NET.Transforms.NormalizeMinMax("Features"))
               .AppendCacheCheckpoint(ML.NET),
            Trainer =
               Trainers.SdcaNonCalibrated(),
            Output =
               ML.NET.Transforms.Conversion.MapKeyToValue("PredictedLabel")
         };
      }
      /// <summary>
      /// Restituisce la previsione
      /// </summary>
      /// <param name="sentence">Significato da prevedere</param>
      /// <returns>La previsione</returns>
      public Prediction GetPrediction(string sentence) => new Prediction( GetPredictionData(null, sentence));
      /// <summary>
      /// Restituisce la previsione
      /// </summary>
      /// <param name="sentence">Significato da prevedere</param>
      /// <returns>Il task della previsione</returns>
      public async Task<Prediction> GetPredictionAsync(string sentence, CancellationToken cancel = default) => new Prediction(await GetPredictionDataAsync(cancel, null, sentence));
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
                  new TextLoader.Column(!string.IsNullOrWhiteSpace(labelColumnName) ? labelColumnName : "Label", DataKind.String, 0),
                  new TextLoader.Column("Sentence", DataKind.String, 1),
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
   /// Risultato della previsione
   /// </summary>
   public sealed partial class TextMeaningRecognizer // Prediction
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
         /// Punteggio per il tipo previsto
         /// </summary>
         public float Score { get; }
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
         internal Prediction(IDataAccess data)
         {
            var grid = data.ToDataViewGrid();
            Meaning = grid[0]["PredictedLabel"];
            var scores = (float[])grid[0]["Score"];
            var slotNames = grid.Schema["Score"].GetSlotNames();
            Scores = slotNames.Zip(scores).Select(item => new KeyValuePair<string, float>(item.First, item.Second)).ToArray();
            Score = Scores.FirstOrDefault(s => s.Key == Meaning).Value;
         }
         #endregion
      }
   }
}
