using Microsoft.ML;
using System;
using System.Diagnostics;

namespace MachineLearning.Model
{
   /// <summary>
   /// Contenitore di pipes di training
   /// </summary>
   public class ModelPipes : IDisposable
   {
      #region Fields
      /// <summary>
      /// Indicatore di oggetto disposed
      /// </summary>
      private bool disposedValue;
      #endregion
      #region Properties
      /// <summary>
      /// Pipe di input e featurizzazione
      /// </summary>
      public IEstimator<ITransformer> Input { get; set; }
      /// <summary>
      /// Pipe unita
      /// </summary>
      public IEstimator<ITransformer> Merged
      {
         get
         {
            var result = (IEstimator<ITransformer>)null;
            if (Input != null)
               result = result == null ? Input : result.Append(Input);
            if (Trainer != null)
               result = result == null ? Trainer : result.Append(Trainer);
            if (Output != null)
               result = result == null ? Output : result.Append(Output);
            return result;
         }
      }
      /// <summary>
      /// Pipe di output
      /// </summary>
      public IEstimator<ITransformer> Output { get; set; }
      /// <summary>
      /// Pipe di training
      /// </summary>
      public IEstimator<ITransformer> Trainer { get; set; }
      #endregion
      #region Methods
      /// <summary>
      /// Implementazione della IDisposable
      /// </summary>
      public void Dispose()
      {
         Dispose(disposing: true);
         GC.SuppressFinalize(this);
      }
      /// <summary>
      /// Funzione di dispose
      /// </summary>
      /// <param name="disposing">Indicatore di dispose da codice</param>
      protected virtual void Dispose(bool disposing)
      {
         if (!disposedValue) {
            foreach (var estimator in new[] { Input, Trainer, Output }) {
               try {
                  (estimator as IDisposable)?.Dispose();
               }
               catch (Exception exc) {
                  Trace.WriteLine(exc);
               }
            }
            Input = null;
            Trainer = null;
            Output = null;
            disposedValue = true;
         }
      }
      #endregion
   }
}
