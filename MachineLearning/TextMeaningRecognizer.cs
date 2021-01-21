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
      /// Schema di input dei dati
      /// </summary>
      public override DataViewSchema InputSchema => base.InputSchema ?? GetDefaultInputSchema();
      /// <summary>
      /// Storage del modello
      /// </summary>
      public IModelStorage ModelStorage { get; set; }
      /// <summary>
      /// Trainer del modello
      /// </summary>
      public IModelTrainer ModelTrainer { get; set; } = new ModelTrainerStandard();
      /// <summary>
      /// Dati di training
      /// </summary>
      public IDataStorage TrainingData { get; set; }
      #endregion
      #region Methods
      /// <summary>
      /// Costruttore
      /// </summary>
      public TextMeaningRecognizer() : base() { }
      /// <summary>
      /// Costruttore
      /// </summary>
      /// <param name="seed">Seme operazioni random</param>
      public TextMeaningRecognizer(int? seed) : base(seed) { }
      /// <summary>
      /// Costruttore
      /// </summary>
      /// <param name="ml">Contesto di machine learning</param>
      public TextMeaningRecognizer(MachineLearningContext ml) : base(ml) { }
      /// <summary>
      /// Aggiunge un elenco di dati di training
      /// </summary>
      /// <param name="data">Dati</param>
      /// <param name="checkForDuplicates">Controllo dei duplicati</param>
      public void AddTrainingData(bool checkForDuplicates, params string[] data) => AddTrainingDataAsync(checkForDuplicates, default, data);
      /// <summary>
      /// Aggiunge un elenco di dati di training
      /// </summary>
      /// <param name="cancellation">Token di cancellazione</param>
      /// <param name="data">Dati</param>
      /// <param name="checkForDuplicates">Controllo dei duplicati</param>
      public Task AddTrainingDataAsync(bool checkForDuplicates, CancellationToken cancellation, params string[] data)
      {
         var dataGrid = DataViewGrid.Create(this, InputSchema);
         dataGrid.Add(data);
         return AddTrainingDataAsync(dataGrid, checkForDuplicates, cancellation);
      }
      /// <summary>
      /// Schema di input dei dati
      /// </summary>
      /// <param name="labelColumnName">Nome colonna label</param>
      /// <param name="dataColumnName">Nome colonna dei dati</param>
      /// <returns>Lo schema di default</returns>
      public static DataViewSchema GetDefaultInputSchema(string labelColumnName = "Label", string dataColumnName = "Data")
      {
         var builder = new DataViewSchema.Builder();
         builder.AddColumn(labelColumnName, TextDataViewType.Instance);
         builder.AddColumn(dataColumnName, TextDataViewType.Instance);
         return builder.ToSchema();
      }
      /// <summary>
      /// Restituisce le opzioni di caricamento in formato testo di default
      /// </summary>
      /// <param name="labelColumnName">Nome colonna label</param>
      /// <param name="dataColumnName">Nome colonna dei dati</param>
      /// <returns>Le opzioni di caricamento testi di default</returns>
      public static TextLoader.Options GetDefaultTextLoaderOptions(string labelColumnName = "Label", string dataColumnName = "Data")
      {
         return new TextLoader.Options
         {
            AllowQuoting = true,
            Separators = new[] { ',' },
            Columns = new[]
            {
               new TextLoader.Column(labelColumnName, DataKind.String, 0),
               new TextLoader.Column(dataColumnName, DataKind.String, 1),
            }
         };
      }
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
               .Append(ML.NET.Transforms.Text.FeaturizeText("FeaturizeText", new TextFeaturizingEstimator.Options(), (from c in InputSchema
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
      /// <param name="sentences">Elenco di sentenze di cui prevedere il significato</param>
      /// <returns>La previsione</returns>
      public Prediction GetPrediction(params string[] sentences) => GetPredictionAsync(default, sentences).ConfigureAwait(false).GetAwaiter().GetResult();
      /// <summary>
      /// Restituisce la previsione
      /// </summary>
      /// <param name="sentences">Elenco di sentenze di cui prevedere il significato</param>
      /// <returns>Il task della previsione</returns>
      public async Task<Prediction> GetPredictionAsync(CancellationToken cancel = default, params string[] sentences)
      {
         var schema = InputSchema;
         var valueIx = 0;
         return new Prediction(await GetPredictionDataAsync(cancel, schema.Select(c => c.Name == LabelColumnName ? "" : sentences[valueIx++]).ToArray()));
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
