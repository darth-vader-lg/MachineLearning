using Microsoft.ML.Runtime;
using System;
using Tensorflow;

namespace MachineLearning.TensorFlow
{
   /// <summary>
   /// Contesto TensorFlow
   /// </summary>
   public class TFContext : tensorflow, IChannelProvider
   {
      #region Fields
      /// <summary>
      /// Sincronizzatore messaggi dei canali
      /// </summary>
      private readonly object channelSync = new();
      #endregion
      #region Properties
      /// <summary>
      /// La stringa descrittiva del contesto stesso
      /// </summary>
      public string ContextDescription => "TensorFlow";
      #endregion
      #region Events
      /// <summary>
      /// Evento di log di un messaggio
      /// </summary>
      public event EventHandler<LoggingEventArgs> Log;
      #endregion
      #region Methods
      /// <summary>
      /// Evento di canale disposto
      /// </summary>
      /// <param name="sender">Il canale</param>
      /// <param name="e">Argomenti dell'evento</param>
      private void ChannelDisposed(object sender, EventArgs e)
      {
         if (sender is Channel ch) {
            ch.Log -= ChannelLog;
            ch.Disposed -= ChannelDisposed;
         }
      }
      /// <summary>
      /// Evento di log da parte di un canale
      /// </summary>
      /// <param name="sender">Canale</param>
      /// <param name="e">Argomenti dell'evento</param>
      private void ChannelLog(object sender, LoggingEventArgs e) => OnChannelLog(e);
      /// <summary>
      /// Funzione di log dei canali
      /// </summary>
      /// <param name="e">Argomenti del log</param>
      protected void OnChannelLog(LoggingEventArgs e)
      {
         lock (channelSync) {
            try {
               Log?.Invoke(this, e);
            }
            catch (Exception) {
            }
         }
      }
      /// <summary>
      /// Processa le eccezioni
      /// </summary>
      /// <typeparam name="TException">Tipo di eccezione</typeparam>
      /// <param name="ex">Exxezione</param>
      /// <returns>L'eccezione</returns>
      public TException Process<TException>(TException ex) where TException : Exception => throw ex;
      /// <summary>
      /// Avvia un canale di messaggi standard
      /// </summary>
      /// <param name="name">Nome del canale</param>
      /// <returns>Il canale</returns>
      public IChannel Start(string name)
      {
         var ch = new Channel(name, ContextDescription);
         ch.Log += ChannelLog;
         ch.Disposed += ChannelDisposed;
         return ch;
      }
      /// <summary>
      /// Avvia una pipe di messaggi generici
      /// </summary>
      /// <typeparam name="TMessage">Tipo di messaggio</typeparam>
      /// <param name="name">Nome della pipe</param>
      /// <returns></returns>
      public IPipe<TMessage> StartPipe<TMessage>(string name) => new MessagePipe<TMessage>(name, ContextDescription);
      #endregion
   }
}
