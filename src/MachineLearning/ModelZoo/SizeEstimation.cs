using MachineLearning.Data;
using MachineLearning.Model;
using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.Diagnostics;
using System.Linq;
using System.Threading;

namespace MachineLearning.ModelZoo
{
   /// <summary>
   /// Classe per la previsione delle taglie
   /// </summary>
   public sealed partial class SizeEstimation :
      ModelZooBase<SizeEstimation.Mdl>,
      IDataStorageProvider,
      IInputSchema,
      IModelStorageProvider,
      IModelTrainerProvider
   {
      #region Properties
      /// <summary>
      /// Storage dei dati
      /// </summary>
      public IDataStorage DataStorage { get; set; }
      /// <summary>
      /// Schema di input
      /// </summary>
      public DataSchema InputSchema { get; private set; }
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
      public SizeEstimation(IContextProvider<MLContext> context = default)
      {
         Model = new Mdl(this, context);
         SetSchema(0, "Size", "Data");
      }
      /// <summary>
      /// Aggiunge un elenco di dati di training
      /// </summary>
      /// <param name="checkForDuplicates">Controllo dei duplicati</param>
      /// <param name="cancellation">Token di cancellazione</param>
      /// <param name="data">Dati</param>
      public void AddTrainingData(bool checkForDuplicates, CancellationToken cancellation, params float[] data)
      {
         var dataGrid = DataViewGrid.Create(Model, InputSchema);
         dataGrid.Add(data);
         Model.AddTrainingData(dataGrid, checkForDuplicates, cancellation);
      }
      /// <summary>
      /// Restituisce la previsione
      /// </summary>
      /// <param name="values">Valori di input</param>
      /// <returns>La previsione</returns>
      public Prediction GetPrediction(CancellationToken cancel = default, params float[] values)
      {
         var schema = InputSchema;
         var valueIx = 0;
         return new Prediction(Model.GetPredictionData(schema.Select(c => (object)(c.Name == Model.LabelColumnName ? 0f : values[valueIx++])).ToArray(), cancel));
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
         Model.LabelColumnName = columnsNames[predictionColumnIndex];
         InputSchema = DataViewSchemaBuilder.Build(columnsNames.Select(c => (c, typeof(float))).ToArray());
      }
      #endregion
   }

   /// <summary>
   /// Modello
   /// </summary>
   public sealed partial class SizeEstimation // Model
   {
      [Serializable]
      public sealed class Mdl :
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
         private readonly SizeEstimation _owner;
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
         IDataStorage IDataStorageProvider.DataStorage => ((IDataStorageProvider)_owner).DataStorage;
         /// <summary>
         /// Schema di input del modello
         /// </summary>
         DataSchema IInputSchema.InputSchema => ((IInputSchema)_owner).InputSchema;
         /// <summary>
         /// Abilitazione salvataggio automatico modello
         /// </summary>
         bool IModelAutoCommit.ModelAutoCommit => true;
         /// <summary>
         /// Abilitazione commit automatico dei dati di training
         /// </summary>
         bool IModelAutoSave.ModelAutoSave => true;
         /// <summary>
         /// Storage del modello
         /// </summary>
         IModelStorage IModelStorageProvider.ModelStorage => ((IModelStorageProvider)_owner).ModelStorage;
         /// <summary>
         /// Trainer del modello
         /// </summary>
         IModelTrainer IModelTrainerProvider.ModelTrainer => ((IModelTrainerProvider)_owner).ModelTrainer;
         /// <summary>
         /// Nome del modello
         /// </summary>
         string IModelName.ModelName => _owner.Name;
         /// <summary>
         /// Opzioni di caricamenti dati in formato testo
         /// </summary>
         TextLoader.Options ITextLoaderOptions.TextLoaderOptions => new()
         {
            Columns = _owner.InputSchema.ToTextLoaderColumns(),
            Separators = new[] { ',' },
         };
         /// <summary>
         /// Storage di dati di training
         /// </summary>
         IDataStorage ITrainingStorageProvider.TrainingStorage { get; } = new DataStorageBinaryMemory();
         #endregion
         #region Methods
         /// <summary>
         /// Costruttore
         /// </summary>
         /// <param name="owner">Oggetto di appartenenza</param>
         /// <param name="context">Contesto di machine learning</param>
         internal Mdl(SizeEstimation owner, IContextProvider<MLContext> context) : base(context) => _owner = owner;
         /// <summary>
         /// Funzione di dispose
         /// </summary>
         /// <param name="disposing">Indicatore di dispose da codice</param>
         protected sealed override void Dispose(bool disposing)
         {
            if (IsDisposed)
               return;
            if (disposing) {
               try {
                  _pipes?.Dispose();
               }
               catch (Exception exc) {
                  Trace.WriteLine(exc);
               }
            }
            _pipes = null;
            base.Dispose(disposing);
         }
         /// <summary>
         /// Restituisce la pipe di training del modello
         /// </summary>
         /// <returns></returns>
         public sealed override ModelPipes GetPipes()
         {
            // Pipe di training
            return _pipes ??= new ModelPipes
            {
               Input =
                  Context.Transforms.Concatenate("Features", (from c in _owner.InputSchema
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
   public sealed partial class SizeEstimation // Prediction
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
