using MachineLearning.Data;
using MachineLearning.Model;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms.Text;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Threading;

namespace MachineLearning.ModelZoo
{
   /// <summary>
   /// Classe per l'interpretazione del significato si testi
   /// </summary>
   [Serializable]
   public sealed partial class TextMeaningRecognizer :
      ModelZooBase<TextMeaningRecognizer.Mdl>,
      IDataStorageProvider,
      IInputSchema,
      IModelStorageProvider,
      IModelTrainerProvider,
      IModelTrainingControl
   {
      #region Properties
      /// <summary>
      /// Storage dei dati
      /// </summary>
      public IDataStorage DataStorage { get; set; }
      /// <summary>
      /// Schema di input dei dati
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
      /// Nome dell'oggetto
      /// </summary>
      public string Name { get; set; }
      #endregion
      #region Methods
      /// <summary>
      /// Costruttore
      /// </summary>
      /// <param name="context">Contesto di machine learning</param>
      public TextMeaningRecognizer(IContextProvider<MLContext> context = default)
      {
         Model = new Mdl(this, context);
         SetSchema(0, "Meaning", "Text");
      }
      /// <summary>
      /// Aggiunge un elenco di dati di training
      /// </summary>
      /// <param name="checkForDuplicates">Controllo dei duplicati</param>
      /// <param name="cancellation">Token di cancellazione</param>
      /// <param name="data">Dati</param>
      public void AddTrainingData(bool checkForDuplicates, CancellationToken cancellation, params string[] data)
      {
         var dataGrid = DataViewGrid.Create(Model, InputSchema);
         dataGrid.Add(data);
         Model.AddTrainingData(dataGrid, checkForDuplicates, cancellation);
      }
      /// <summary>
      /// Pulisce il modello
      /// </summary>
      public void ClearModel() => Model.ClearModel();
      /// <summary>
      /// Restituisce la previsione
      /// </summary>
      /// <param name="cancel">L'eventuale tokehn di cancellazione</param>
      /// <param name="sentences">Elenco di sentenze di cui prevedere il significato</param>
      /// <returns>La previsione</returns>
      public Prediction GetPrediction(CancellationToken cancel = default, params string[] sentences)
      {
         var schema = InputSchema;
         var valueIx = 0;
         return new Prediction(Model.GetPredictionData(schema.Select(c => c.Name == Model.LabelColumnName ? "" : sentences[valueIx++]).ToArray(), cancel));
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
         InputSchema = DataViewSchemaBuilder.Build(columnsNames.Select(c => (c, typeof(string))).ToArray());
      }
      #endregion
   }

   /// <summary>
   /// Modello
   /// </summary>
   public sealed partial class TextMeaningRecognizer // Mdl
   {
      [Serializable]
      public sealed class Mdl :
         MulticlassModelBase,
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
         private readonly TextMeaningRecognizer _owner;
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
         IDataStorage IDataStorageProvider.DataStorage => _owner.DataStorage;
         /// <summary>
         /// Schema di input del modello
         /// </summary>
         DataViewSchema IInputSchema.InputSchema => _owner.InputSchema;
         /// <summary>
         /// Abilitazione salvataggio automatico modello
         /// </summary>
         bool IModelAutoCommit.ModelAutoCommit => true;
         /// <summary>
         /// Abilitazione commit automatico dei dati di training
         /// </summary>
         bool IModelAutoSave.ModelAutoSave => true;
         /// <summary>
         /// Nome del modello
         /// </summary>
         string IModelName.ModelName => _owner.Name;
         /// <summary>
         /// Storage del modello
         /// </summary>
         IModelStorage IModelStorageProvider.ModelStorage => _owner.ModelStorage;
         /// <summary>
         /// Trainer del modello
         /// </summary>
         IModelTrainer IModelTrainerProvider.ModelTrainer => _owner.ModelTrainer;
         /// <summary>
         /// Opzioni di caricamento dati in formato testo
         /// </summary>
         TextLoader.Options ITextLoaderOptions.TextLoaderOptions => new()
         {
            AllowQuoting = true,
            Separators = new[] { ',' },
            Columns = _owner.InputSchema.ToTextLoaderColumns(),
         };
         /// <summary>
         /// Dati di training
         /// </summary>
         IDataStorage ITrainingStorageProvider.TrainingStorage { get; } = new DataStorageBinaryMemory();
         #endregion
         #region Methods
         /// <summary>
         /// Costruttore
         /// </summary>
         /// <param name="owner">Oggetto di appartenenza</param>
         /// <param name="context">Contesto di machine learning</param>
         internal Mdl(TextMeaningRecognizer owner, IContextProvider<MLContext> context) : base(context) => _owner = owner;
         /// <summary>
         /// Funzione di dispose
         /// </summary>
         /// <param name="disposing">Indicatore di dispose da codice</param>
         protected override void Dispose(bool disposing)
         {
            base.Dispose(disposing);
            try {
               _pipes?.Dispose();
            }
            catch (Exception exc) {
               Trace.WriteLine(exc);
            }
            _pipes = null;
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
                  Context.Transforms.Conversion.MapValueToKey(LabelColumnName)
                  .Append(Context.Transforms.Text.FeaturizeText("FeaturizeText", new TextFeaturizingEstimator.Options(), (from c in _owner.InputSchema
                                                                                                                          where c.Name != LabelColumnName
                                                                                                                          select c.Name).ToArray()))
                  .Append(Context.Transforms.CopyColumns("Features", "FeaturizeText"))
                  .Append(Context.Transforms.NormalizeMinMax("Features"))
                  .AppendCacheCheckpoint(Context),
               Trainer =
                  Trainers.SdcaNonCalibrated(new Microsoft.ML.Trainers.SdcaNonCalibratedMulticlassTrainer.Options
                  {
                     LabelColumnName = LabelColumnName
                  }),
               Output =
                  Context.Transforms.Conversion.MapKeyToValue("PredictedLabel")
            };
         }
         #endregion
      }
   }

   /// <summary>
   /// Risultato della previsione
   /// </summary>
   public sealed partial class TextMeaningRecognizer // Prediction
   {
      [Serializable]
      public sealed class Prediction
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
