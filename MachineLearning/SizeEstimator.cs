using MachineLearning.Data;
using MachineLearning.Model;
using MachineLearning.Util;
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
   public sealed partial class SizeEstimator :
      IDataStorageProvider,
      IInputSchema,
      IModelStorageProvider,
      IModelTrainerProvider
   {
      #region Fields
      /// <summary>
      /// Modello
      /// </summary>
      private readonly Model _model;
      #endregion
      #region Properties
      /// <summary>
      /// Storage dei dati
      /// </summary>
      public IDataStorage DataStorage { get; set; }
      /// <summary>
      /// Schema di input
      /// </summary>
      public DataViewSchema InputSchema { get; private set; }
      /// <summary>
      /// Storage del modello
      /// </summary>
      public IModelStorage ModelStorage { get; set; }
      /// <summary>
      /// Trainer del modello
      /// </summary>
      public IModelTrainer ModelTrainer { get; set; } = new ModelTrainerStandard();
      /// <summary>
      /// Nome del modello
      /// </summary>
      public string Name { get; set; }
      /// <summary>
      /// Dati di training
      /// </summary>
      public IDataStorage TrainingData { get; set; } = new DataStorageBinaryMemory();
      #endregion
      #region Methods
      /// <summary>
      /// Costruttore
      /// </summary>
      /// <param name="context">Contesto di machine learning</param>
      public SizeEstimator(IContextProvider<MLContext> context = default)
      {
         _model = new Model(this, context);
         SetSchema(0, "Size", "Data");
      }
      /// <summary>
      /// Aggiunge un elenco di dati di training
      /// </summary>
      /// <param name="checkForDuplicates">Controllo dei duplicati</param>
      /// <param name="data">Dati</param>
      public void AddTrainingData(bool checkForDuplicates, params float[] data) => AddTrainingDataAsync(checkForDuplicates, default, data).WaitSync();
      /// <summary>
      /// Aggiunge un elenco di dati di training
      /// </summary>
      /// <param name="checkForDuplicates">Controllo dei duplicati</param>
      /// <param name="cancellation">Token di cancellazione</param>
      /// <param name="data">Dati</param>
      public Task AddTrainingDataAsync(bool checkForDuplicates, CancellationToken cancellation, params float[] data)
      {
         var dataGrid = DataViewGrid.Create(_model, _model.InputSchema);
         dataGrid.Add(data);
         return _model.AddTrainingDataAsync(dataGrid, checkForDuplicates, cancellation);
      }
      /// <summary>
      /// Restituisce la previsione
      /// </summary>
      /// <param name="values">Valori di input</param>
      /// <returns>La previsione</returns>
      public Prediction GetPrediction(params float[] values) => GetPredictionAsync(default, values).WaitSync();
      /// <summary>
      /// Restituisce la previsione
      /// </summary>
      /// <param name="values">Valori di input</param>
      /// <returns>Il task della previsione</returns>
      public async Task<Prediction> GetPredictionAsync(CancellationToken cancel = default, params float[] values)
      {
         var schema = InputSchema;
         var valueIx = 0;
         return new Prediction(await _model.GetPredictionDataAsync(schema.Select(c => (object)(c.Name == _model.LabelColumnName ? 0f : values[valueIx++])).ToArray(), cancel));
      }
      /// <summary>
      /// Imposta lo schema dei dati
      /// </summary>
      /// <param name="predictionColumnIndex">Indice della colonna di previsione</param>
      /// <param name="columnsNames">Nomi delle colonne dello schema</param>
      public void SetSchema(int predictionColumnIndex = 0, params string[] columnsNames)
      {
         if (predictionColumnIndex < 0 || predictionColumnIndex >= columnsNames.Length)
            throw new ArgumentException("The prediction column index is out of range", nameof(predictionColumnIndex));
         if (columnsNames.Any(item => string.IsNullOrEmpty(item)))
            throw new ArgumentException("All the columns must have a name", nameof(columnsNames));
         _model.LabelColumnName = columnsNames[predictionColumnIndex];
         InputSchema = DataViewSchemaBuilder.Build(columnsNames.Select(c => (c, typeof(float))).ToArray());
      }
      #endregion
   }

   /// <summary>
   /// Modello
   /// </summary>
   public sealed partial class SizeEstimator // Model
   {
      [Serializable]
      private sealed class Model :
         RegressionModelBase,
         IDataStorageProvider,
         IInputSchema,
         IModelAutoCommit,
         IModelAutoSave,
         IModelName,
         IModelStorageProvider,
         IModelTrainerProvider,
         ITextLoaderOptions,
         ITrainingStorageProvider
      {
         #region Fields
         /// <summary>
         /// Oggetto di appartenenza
         /// </summary>
         private readonly SizeEstimator _owner;
         /// <summary>
         /// Pipe di training
         /// </summary>
         [NonSerialized]
         private ModelPipes _pipes;
         #endregion
         #region Properties
         /// <summary>
         /// Storage di dati
         /// </summary>
         public IDataStorage DataStorage => ((IDataStorageProvider)_owner).DataStorage;
         /// <summary>
         /// Schema di input del modello
         /// </summary>
         public DataViewSchema InputSchema => ((IInputSchema)_owner).InputSchema;
         /// <summary>
         /// Abilitazione salvataggio automatico modello
         /// </summary>
         public bool ModelAutoCommit => true;
         /// <summary>
         /// Abilitazione commit automatico dei dati di training
         /// </summary>
         public bool ModelAutoSave => true;
         /// <summary>
         /// Storage del modello
         /// </summary>
         public IModelStorage ModelStorage => ((IModelStorageProvider)_owner).ModelStorage;
         /// <summary>
         /// Trainer del modello
         /// </summary>
         public IModelTrainer ModelTrainer => ((IModelTrainerProvider)_owner).ModelTrainer;
         /// <summary>
         /// Nome del modello
         /// </summary>
         public string ModelName => _owner.Name;
         /// <summary>
         /// Opzioni di caricamenti dati in formato testo
         /// </summary>
         public TextLoader.Options TextLoaderOptions => new()
         {
            Columns = InputSchema.ToTextLoaderColumns(),
            Separators = new[] { ',' },
         };
         /// <summary>
         /// Storage di dati di training
         /// </summary>
         public IDataStorage TrainingStorage { get; } = new DataStorageBinaryMemory();
         #endregion
         #region Methods
         /// <summary>
         /// Costruttore
         /// </summary>
         /// <param name="owner">Oggetto di appartenenza</param>
         /// <param name="context">Contesto di machine learning</param>
         internal Model(SizeEstimator owner, IContextProvider<MLContext> context) : base(context) => _owner = owner;
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
                  Context.Transforms.Concatenate("Features", (from c in InputSchema
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
   }

   /// <summary>
   /// Risultato della previsione
   /// </summary>
   public sealed partial class SizeEstimator // Prediction
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
