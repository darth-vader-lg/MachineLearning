using MachineLearning.Model;
using System;
using System.Diagnostics;
using System.Threading;

namespace MachineLearning.ModelZoo
{
   /// <summary>
   /// Base class for model zoo
   /// </summary>
   [Serializable]
   public abstract class ModelZooBase<TModel> : IDisposable where TModel: ModelBase, IDisposable, IModelTrainingControl
   {
      #region Fields
      /// <summary>
      /// Disposed object
      /// </summary>
      [NonSerialized]
      private bool disposedValue;
      /// <summary>
      /// The associated model
      /// </summary>
      private TModel model;
      #endregion
      #region Properties
      /// <summary>
      /// Modello
      /// </summary>
      protected internal TModel Model
      {
         get => model;
         set
         {
            if (model != null) {
               model.ModelChanged -= Model_ModelChanged;
               model.TrainingCycleStarted -= Model_TrainingCycleStarted;
               model.TrainingDataChanged -= Model_TrainingDataChanged;
               model.TrainingEnded -= Model_TrainingEnded;
               model.TrainingStarted -= Model_TrainingStarted;
            }
            model = value;
            if (model != null) {
               model.ModelChanged += Model_ModelChanged;
               model.TrainingCycleStarted += Model_TrainingCycleStarted;
               model.TrainingDataChanged += Model_TrainingDataChanged;
               model.TrainingEnded += Model_TrainingEnded;
               model.TrainingStarted += Model_TrainingStarted;
            }
         }
      }
      /// <summary>
      /// Enable the event syncronization with the creation context
      /// </summary>
      public bool SyncEvents { get; set; }
      #endregion
      #region Events
      /// <summary>
      /// Model changed event
      /// </summary>
      [field: NonSerialized] 
      public event ModelTrainingEventHandler ModelChanged;
      /// <summary>
      /// Train started event
      /// </summary>
      [field: NonSerialized]
      public event ModelTrainingEventHandler TrainingCycleStarted;
      /// <summary>
      /// Train data changed event
      /// </summary>
      [field: NonSerialized]
      public event ModelTrainingEventHandler TrainingDataChanged;
      /// <summary>
      /// Train ended event
      /// </summary>
      [field: NonSerialized]
      public event ModelTrainingEventHandler TrainingEnded;
      /// <summary>
      /// Train started event
      /// </summary>
      [field: NonSerialized]
      public event ModelTrainingEventHandler TrainingStarted;
      #endregion
      #region Methods
      /// <summary>
      /// IDisposable implementation
      /// </summary>
      public void Dispose()
      {
         Dispose(disposing: true);
         GC.SuppressFinalize(this);
      }
      /// <summary>
      /// Dispose function
      /// </summary>
      /// <param name="disposing"></param>
      protected virtual void Dispose(bool disposing)
      {
         if (!disposedValue) {
            try {
               Model?.Dispose();
            }
            catch (Exception exc) {
               Trace.WriteLine(exc);
            }
            disposedValue = true;
         }
      }
      /// <summary>
      /// Redirect model changed event
      /// </summary>
      /// <param name="sender">The model sender</param>
      /// <param name="e">Arguments</param>
      private void Model_ModelChanged(object sender, ModelTrainingEventArgs e) => OnModelChanged(e);
      /// <summary>
      /// Redirect train cycle started event
      /// </summary>
      /// <param name="sender">The model sender</param>
      /// <param name="e">Arguments</param>
      private void Model_TrainingCycleStarted(object sender, ModelTrainingEventArgs e) => OnTrainingCycleStarted(e);
      /// <summary>
      /// Redirect train data changed event
      /// </summary>
      /// <param name="sender">The model sender</param>
      /// <param name="e">Arguments</param>
      private void Model_TrainingDataChanged(object sender, ModelTrainingEventArgs e) => OnTrainingDataChanged(e);
      /// <summary>
      /// Redirect train ended event
      /// </summary>
      /// <param name="sender">The model sender</param>
      /// <param name="e">Arguments</param>
      private void Model_TrainingEnded(object sender, ModelTrainingEventArgs e) => OnTrainingEnded(e);
      /// <summary>
      /// Redirect train started event
      /// </summary>
      /// <param name="sender">The model sender</param>
      /// <param name="e">Arguments</param>
      private void Model_TrainingStarted(object sender, ModelTrainingEventArgs e) => OnTrainingStarted(e);
      /// <summary>
      /// Model changed function
      /// </summary>
      /// <param name="e">Arguments</param>
      protected void OnModelChanged(ModelTrainingEventArgs e)
      {
         try {
            if (SyncEvents && Model?.SyncronizationContext != null && ModelChanged != null)
               Model.SyncronizationContext.SyncExec(() => ModelChanged?.Invoke(this, e));
            else
               ModelChanged?.Invoke(this, e);
         }
         catch (Exception exc) {
            Trace.WriteLine(exc);
         }
      }
      /// <summary>
      /// Train cycle started function
      /// </summary>
      /// <param name="e">Arguments</param>
      protected void OnTrainingCycleStarted(ModelTrainingEventArgs e)
      {
         try {
            if (SyncEvents && Model?.SyncronizationContext != null && TrainingCycleStarted != null)
               Model.SyncronizationContext.SyncExec(() => TrainingCycleStarted?.Invoke(this, e));
            else
               TrainingCycleStarted?.Invoke(this, e);
         }
         catch (Exception exc) {
            Trace.WriteLine(exc);
         }
      }
      /// <summary>
      /// Train data changed function
      /// </summary>
      /// <param name="e">Arguments</param>
      protected void OnTrainingDataChanged(ModelTrainingEventArgs e)
      {
         try {
            if (SyncEvents && Model?.SyncronizationContext != null && TrainingDataChanged != null)
               Model.SyncronizationContext.SyncExec(() => TrainingDataChanged?.Invoke(this, e));
            else
               TrainingDataChanged?.Invoke(this, e);
         }
         catch (Exception exc) {
            Trace.WriteLine(exc);
         }
      }
      /// <summary>
      /// Train ended function
      /// </summary>
      /// <param name="e">Arguments</param>
      protected void OnTrainingEnded(ModelTrainingEventArgs e)
      {
         try {
            if (SyncEvents && Model?.SyncronizationContext != null && TrainingEnded != null)
               Model.SyncronizationContext.SyncExec(() => TrainingEnded?.Invoke(this, e));
            else
               TrainingEnded?.Invoke(this, e);
         }
         catch (Exception exc) {
            Trace.WriteLine(exc);
         }
      }
      /// <summary>
      /// Train started function
      /// </summary>
      /// <param name="e">Arguments</param>
      protected void OnTrainingStarted(ModelTrainingEventArgs e)
      {
         try {
            if (SyncEvents && Model?.SyncronizationContext != null && TrainingStarted != null)
               Model.SyncronizationContext.SyncExec(() => TrainingStarted?.Invoke(this, e));
            else
               TrainingStarted?.Invoke(this, e);
         }
         catch (Exception exc) {
            Trace.WriteLine(exc);
         }
      }
      /// <summary>
      /// Start the train of the model
      /// </summary>
      /// <param name="cancellation">Eventuale token di cancellazione del training</param>
      public void StartTraining(CancellationToken cancellation = default) => Model?.StartTraining(cancellation);
      /// <summary>
      /// Stop the train of the model
      /// </summary>
      /// <param name="cancellation">Eventuale token di cancellazione dell'attesa</param>
      public void StopTraining(CancellationToken cancellation = default) => Model?.StopTraining(cancellation);
      /// <summary>
      /// Wait for model changed
      /// </summary>
      /// <param name="cancellation">Optional cancellation token</param>
      public void WaitModelChanged(CancellationToken cancellation = default)
      {
         ManualResetEvent eventWaitHandle;
         void WaitEvent(object sender, ModelTrainingEventArgs e) => eventWaitHandle.Set();
         using (eventWaitHandle = new ManualResetEvent(false)) {
            try {
               ModelChanged += WaitEvent;
               while (!eventWaitHandle.WaitOne(500))
                  cancellation.ThrowIfCancellationRequested();
               cancellation.ThrowIfCancellationRequested();
            }
            finally {
               ModelChanged -= WaitEvent;
            }
         }
      }
      /// <summary>
      /// Wait for train ended
      /// </summary>
      /// <param name="cancellation">Optional cancellation token</param>
      public void WaitTrainingEnded(CancellationToken cancellation = default)
      {
         ManualResetEvent eventWaitHandle;
         void WaitEvent(object sender, ModelTrainingEventArgs e) => eventWaitHandle.Set();
         using (eventWaitHandle = new ManualResetEvent(false)) {
            try {
               TrainingEnded += WaitEvent;
               while (!eventWaitHandle.WaitOne(500))
                  cancellation.ThrowIfCancellationRequested();
               cancellation.ThrowIfCancellationRequested();
            }
            finally {
               TrainingEnded -= WaitEvent;
            }
         }
      }
      /// <summary>
      /// Wait for singled train cycle started
      /// </summary>
      /// <param name="cancellation">Optional cancellation token</param>
      public void WaitTrainingCycleStarted(CancellationToken cancellation = default)
      {
         ManualResetEvent eventWaitHandle;
         void WaitEvent(object sender, ModelTrainingEventArgs e) => eventWaitHandle.Set();
         using (eventWaitHandle = new ManualResetEvent(false)) {
            try {
               TrainingCycleStarted += WaitEvent;
               while (!eventWaitHandle.WaitOne(500))
                  cancellation.ThrowIfCancellationRequested();
               cancellation.ThrowIfCancellationRequested();
            }
            finally {
               TrainingCycleStarted -= WaitEvent;
            }
         }
      }
      /// <summary>
      /// Wait for global train loop started
      /// </summary>
      /// <param name="cancellation">Optional cancellation token</param>
      public void WaitTrainingStarted(CancellationToken cancellation = default)
      {
         ManualResetEvent eventWaitHandle;
         void WaitEvent(object sender, ModelTrainingEventArgs e) => eventWaitHandle.Set();
         using (eventWaitHandle = new ManualResetEvent(false)) {
            try {
               TrainingStarted += WaitEvent;
               while (!eventWaitHandle.WaitOne(500))
                  cancellation.ThrowIfCancellationRequested();
               cancellation.ThrowIfCancellationRequested();
            }
            finally {
               TrainingStarted -= WaitEvent;
            }
         }
      }
      #endregion
   }
}
