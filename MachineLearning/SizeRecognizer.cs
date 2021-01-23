using MachineLearning.Data;
using MachineLearning.Model;
using Microsoft.ML;
using Microsoft.ML.Data;
using System;
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
      /// Opzioni di caricamenti dati in formato testo
      /// </summary>
      public TextLoader.Options TextLoaderOptions => new TextLoader.Options
      {
         Columns = InputSchema.ToTextLoaderColumns(),
         Separators = new[] { ',' },
      };
      /// <summary>
      /// Dati di training
      /// </summary>
      public IDataStorage TrainingData { get; set; }
      #endregion
      #region Methods
      /// <summary>
      /// Costruttore
      /// </summary>
      /// <param name="ml">Contesto di machine learning</param>
      public SizeRecognizer(MachineLearningContext ml = default) : base(ml) =>
         InputSchema = DataViewSchemaBuilder.Build((LabelColumnName, typeof(string)), ("Data", typeof(float)));
      /// <summary>
      /// Aggiunge un elenco di dati di training
      /// </summary>
      /// <param name="data">Dati</param>
      /// <param name="checkForDuplicates">Controllo dei duplicati</param>
      public void AddTrainingData(bool checkForDuplicates, params float[] data) => AddTrainingDataAsync(checkForDuplicates, default, data);
      /// <summary>
      /// Aggiunge un elenco di dati di training
      /// </summary>
      /// <param name="cancellation">Token di cancellazione</param>
      /// <param name="data">Dati</param>
      /// <param name="checkForDuplicates">Controllo dei duplicati</param>
      public Task AddTrainingDataAsync(bool checkForDuplicates, CancellationToken cancellation, params float[] data)
      {
         var dataGrid = DataViewGrid.Create(this, InputSchema);
         dataGrid.Add(data);
         return AddTrainingDataAsync(dataGrid, checkForDuplicates, cancellation);
      }
      /// <summary>
      /// Restituisce la previsione
      /// </summary>
      /// <param name="values">Valori di input</param>
      /// <returns>La previsione</returns>
      public Prediction GetPrediction(params float[] values) => GetPredictionAsync(default, values).ConfigureAwait(false).GetAwaiter().GetResult();
      /// <summary>
      /// Restituisce la previsione
      /// </summary>
      /// <param name="values">Valori di input</param>
      /// <returns>Il task della previsione</returns>
      public async Task<Prediction> GetPredictionAsync(CancellationToken cancel = default, params float[] values)
      {
         var schema = InputSchema;
         var valueIx = 0;
         return new Prediction(await GetPredictionDataAsync(schema.Select(c => (object)(c.Name == LabelColumnName ? 0f : values[valueIx++])).ToArray(), cancel));
      }
      /// <summary>
      /// Restituisce la pipe di training del modello
      /// </summary>
      /// <returns></returns>
      public override ModelPipes GetPipes()
      {
         // Pipe di training
         return _pipes ??= new ModelPipes
         {
            Input =
               ML.NET.Transforms.Concatenate("Features", (from c in InputSchema
                                                          where c.Name != LabelColumnName
                                                          select c.Name).ToArray()),
            Trainer =
               Trainers.LightGbm(new Microsoft.ML.Trainers.LightGbm.LightGbmRegressionTrainer.Options
               {
                  LabelColumnName = LabelColumnName,
               }),
         };
      }
      #endregion
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
