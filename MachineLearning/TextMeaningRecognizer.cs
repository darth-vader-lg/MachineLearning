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
      IInputSchemaProvider,
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
      /// Schema di input dei dati
      /// </summary>
      public DataViewSchema InputSchema { get; set; }
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
      /// <summary>
      /// Opzioni di caricamento dati in formato testo
      /// </summary>
      public TextLoader.Options TextLoaderOptions => new TextLoader.Options
      {
         AllowQuoting = true,
         Separators = new[] { ',' },
         Columns = InputSchema.ToTextLoaderColumns(),
      };
      #endregion
      #region Methods
      /// <summary>
      /// Costruttore
      /// </summary>
      /// <param name="ml">Contesto di machine learning</param>
      public TextMeaningRecognizer(MachineLearningContext ml = default) : base(ml) =>
         InputSchema = DataViewSchemaBuilder.Build((LabelColumnName, typeof(string)), ("Data", typeof(string)));
      /// <summary>
      /// Aggiunge un elenco di dati di training
      /// </summary>
      /// <param name="data">Dati</param>
      /// <param name="checkForDuplicates">Controllo dei duplicati</param>
      public void AddTrainingData(bool checkForDuplicates, params string[] data) => AddTrainingDataAsync(checkForDuplicates, default, data).ConfigureAwait(false).GetAwaiter().GetResult();
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
         return new Prediction(await GetPredictionDataAsync(schema.Select(c => c.Name == LabelColumnName ? "" : sentences[valueIx++]).ToArray(), cancel));
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
