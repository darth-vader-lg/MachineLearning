using MachineLearning.Data;
using Microsoft.ML;
using System;
using System.Threading;
using System.Threading.Tasks;

namespace MachineLearning.Model
{
   /// <summary>
   /// Interfaccia per gli evaluator
   /// </summary>
   public interface IModelEvaluator
   {
      #region Properties
      /// <summary>
      /// Valutazione disponibile
      /// </summary>
      WaitHandle Available { get; }
      /// <summary>
      /// Token di cancellazione
      /// </summary>
      CancellationToken Cancellation { get; }
      /// <summary>
      /// Storage di dati
      /// </summary>
      IDataStorage DataStorage { get; }
      /// <summary>
      /// Schema di input
      /// </summary>
      DataViewSchema InputSchema { get; }
      /// <summary>
      /// Modello
      /// </summary>
      IDataTransformer Model { get; }
      /// <summary>
      /// Abilitazione al commit automatico dei dati di training
      /// </summary>
      bool ModelAutoCommit { get; }
      /// <summary>
      /// Abilitazione al salvataggio automatico del modello
      /// </summary>
      bool ModelAutoSave { get; }
      /// <summary>
      /// Storage di dati
      /// </summary>
      IModelStorage ModelStorage { get; }
      /// <summary>
      /// Task di training
      /// </summary>
      Task Task { get; }
      /// <summary>
      /// Data e ora dell'evaluator
      /// </summary>
      DateTime Timestamp { get; }
      /// <summary>
      /// Tipo di trainer da utilizzare
      /// </summary>
      IModelTrainer Trainer { get; }
      /// <summary>
      /// Storage di dati di training
      /// </summary>
      IDataStorage TrainingStorage { get; }
      /// <summary>
      /// Contatore di training
      /// </summary>
      int TrainsCount { get; }
      #endregion
      #region Methods
      /// <summary>
      /// Funzione di cancellazione del task di valutazione
      /// </summary>
      void Cancel();
      #endregion
   }
}
